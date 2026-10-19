
from pathlib import Path
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras import layers

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
)


# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Chest X-ray Research Assistant",
    page_icon="🩻",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "chest_xray_3class_best.keras"

CLASS_NAMES = ["Normal", "Tuberculosis", "COVID-19"]
IMAGE_SIZE = (224, 224)

HISTORY_CSV = (
    PROJECT_DIR
    / "patient_history_comparison"
    / "synthetic_record_comparison.csv"
)

UNCERTAINTY_CSV = (
    PROJECT_DIR
    / "uncertainty_analysis"
    / "test_uncertainty_review_results.csv"
)

REVIEW_SUMMARY_CSV = (
    PROJECT_DIR
    / "uncertainty_analysis"
    / "review_flag_summary.csv"
)

EVALUATION_CSV = (
    PROJECT_DIR
    / "evaluation_results"
    / "covid_lung_opacity_errors.csv"
)

REPORT_JSON = PROJECT_DIR / "ai_reports" / "research_report_image_0.json"
REPORT_TXT = PROJECT_DIR / "ai_reports" / "research_report_image_0.txt"


# =========================================================
# CUSTOM KERAS LAYER AND MODEL LOADER
# =========================================================
@tf.keras.utils.register_keras_serializable(package="ThreeClass")
class AddPositionEmbedding3(layers.Layer):
    def build(self, input_shape):
        self.num_tokens = input_shape[1]
        self.embedding_dim = input_shape[2]

        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(self.num_tokens, self.embedding_dim),
            initializer="random_normal",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        return inputs + tf.expand_dims(
            self.position_embeddings, axis=0
        )


@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find the trained model:\n{MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "AddPositionEmbedding3": AddPositionEmbedding3,
            "ThreeClass>AddPositionEmbedding3": AddPositionEmbedding3,
        },
        compile=False,
    )


def build_gradcam_model(model):
    """
    Expose the CNN feature tensor and the original predictions.
    Grad-CAM is only displayed if gradients are connected.
    """
    cnn_features = model.get_layer("reshape").input

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[cnn_features, model.outputs[0]],
        name="connected_gradcam_model",
    )


# =========================================================
# IMAGE PREPROCESSING AND PREDICTION
# =========================================================
def prepare_image(pil_image):
    """
    Match the project's documented preprocessing:
    RGB, resize to 224x224, float32, pixel range 0-255.
    """
    rgb_image = pil_image.convert("RGB")
    array = np.asarray(rgb_image, dtype=np.float32)

    resized = tf.image.resize(array, IMAGE_SIZE)
    batch = tf.expand_dims(resized, axis=0)

    return batch, np.clip(resized.numpy(), 0, 255).astype(np.uint8)


def calculate_uncertainty(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.clip(scores, 1e-12, 1.0)
    scores = scores / scores.sum()

    entropy = -np.sum(scores * np.log(scores))
    normalized_entropy = entropy / np.log(len(scores))

    sorted_scores = np.sort(scores)
    margin = float(sorted_scores[-1] - sorted_scores[-2])

    max_score = float(np.max(scores))

    # Exploratory project rule, not a medical triage rule.
    review_flag = bool(
        max_score < 0.80 or normalized_entropy > 0.40
    )

    return {
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "margin": margin,
        "max_score": max_score,
        "review_flag": review_flag,
    }


# =========================================================
# GRAD-CAM
# =========================================================
def generate_gradcam(model, input_batch, display_image):
    """
    Return a heatmap and overlay only when gradients are valid.
    Returns (None, None, error_message) if generation fails.
    """
    try:
        grad_model = build_gradcam_model(model)

        with tf.GradientTape() as tape:
            feature_maps, predictions = grad_model(
                input_batch, training=False
            )

            class_index = tf.argmax(predictions[0])
            class_score = predictions[:, class_index]

        gradients = tape.gradient(class_score, feature_maps)

        if gradients is None:
            return (
                None,
                None,
                "Gradients were disconnected. No heatmap was displayed.",
            )

        weights = tf.reduce_mean(
            gradients, axis=(1, 2), keepdims=True
        )

        heatmap = tf.reduce_sum(
            weights * feature_maps, axis=-1
        )[0]

        heatmap = tf.maximum(heatmap, 0)
        maximum = tf.reduce_max(heatmap)

        if float(maximum.numpy()) <= 1e-12:
            return None, None, "The Grad-CAM activation map was empty."

        heatmap = heatmap / maximum

        heatmap = tf.image.resize(
            heatmap[..., tf.newaxis], IMAGE_SIZE
        ).numpy().squeeze()

        # Create overlay in memory for display and PDF export.
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(display_image)
        ax.imshow(
            heatmap,
            cmap="jet",
            alpha=0.40,
            vmin=0,
            vmax=1,
        )
        ax.axis("off")
        ax.set_title("Grad-CAM overlay")
        fig.tight_layout()

        overlay_buffer = BytesIO()
        fig.savefig(
            overlay_buffer,
            format="png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)
        overlay_buffer.seek(0)

        return heatmap, overlay_buffer.getvalue(), None

    except Exception as exc:
        return None, None, f"Grad-CAM generation failed: {exc}"


# =========================================================
# PDF REPORT
# =========================================================
def create_pdf_report(
    predicted_class,
    scores,
    uncertainty,
    image_name,
    gradcam_png=None,
):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(
        Paragraph(
            "Chest X-ray Classification Research Report",
            styles["Title"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Experimental academic research output — "
            "not a medical diagnosis.",
            styles["Warning"],
        )
        if "Warning" in styles
        else Paragraph(
            "Experimental academic research output — "
            "not a medical diagnosis.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            f"Image filename: {image_name}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("Model output", styles["Heading2"]))
    story.append(
        Paragraph(
            f"Predicted class: <b>{predicted_class}</b>",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 8))

    table_data = [["Class", "Model score"]]
    for class_name, score in zip(CLASS_NAMES, scores):
        table_data.append([class_name, f"{float(score) * 100:.2f}%"])

    score_table = Table(table_data, colWidths=[3.2 * inch, 1.7 * inch])
    score_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(score_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Uncertainty indicators", styles["Heading2"]))
    story.append(
        Paragraph(
            f"Normalized entropy: {uncertainty['normalized_entropy']:.4f}",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            f"Top-two score margin: {uncertainty['margin']:.4f}",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            f"Maximum model score: {uncertainty['max_score']:.4f}",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            "Experimental review flag: "
            + ("REVIEW FLAGGED" if uncertainty["review_flag"] else "NOT FLAGGED"),
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 14))

    if gradcam_png:
        story.append(Paragraph("Grad-CAM visualization", styles["Heading2"]))
        story.append(
            Paragraph(
                "This visualization reflects model gradients and activations. "
                "It does not prove that highlighted regions are medically meaningful.",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 8))

        try:
            story.append(
                PDFImage(
                    BytesIO(gradcam_png),
                    width=5.5 * inch,
                    height=4.0 * inch,
                )
            )
        except Exception:
            story.append(
                Paragraph(
                    "The Grad-CAM image could not be embedded in this PDF.",
                    styles["BodyText"],
                )
            )

    story.append(Spacer(1, 14))
    story.append(Paragraph("Limitations", styles["Heading2"]))
    story.append(
        Paragraph(
            "This model is an experimental research prototype. Scores are "
            "not necessarily calibrated probabilities. Results may be wrong "
            "and must not be used to diagnose disease, select treatment, or "
            "replace assessment by a qualified healthcare professional. "
            "Dataset splitting, patient-level independence, external validation, "
            "and clinical validity must be assessed separately.",
            styles["BodyText"],
        )
    )

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# UTILITIES FOR EXISTING PROJECT FILES
# =========================================================
def show_csv_preview(title, path, max_rows=10):
    st.subheader(title)

    if not path.exists():
        st.info(f"File not found: {path.relative_to(PROJECT_DIR)}")
        return

    try:
        dataframe = pd.read_csv(path)
        st.caption(
            f"{path.relative_to(PROJECT_DIR)} — "
            f"{len(dataframe)} rows, {len(dataframe.columns)} columns"
        )
        st.dataframe(dataframe.head(max_rows), use_container_width=True)

        st.download_button(
            label=f"Download {path.name}",
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
            key=f"download_{path.name}",
        )
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")


def load_csv_if_available(path):
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


# =========================================================
# SESSION STATE
# =========================================================
for key, default in {
    "prediction_result": None,
    "original_image_bytes": None,
    "gradcam_png": None,
    "gradcam_error": None,
    "uncertainty_result": None,
    "image_filename": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# SIDEBAR AND APP NAVIGATION
# =========================================================
st.sidebar.title("🩻 Research Assistant")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "X-ray Classification",
        "Grad-CAM Explainability",
        "Uncertainty Analysis",
        "Patient History Comparison",
        "Research Reports",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Hybrid CNN–Transformer")
st.sidebar.caption("Three-class research prototype")


# =========================================================
# GLOBAL SAFETY NOTICE
# =========================================================
st.title("🩻 Chest X-ray Research Assistant")

st.warning(
    "RESEARCH USE ONLY — This application is not a medical diagnostic "
    "tool. Model predictions and heatmaps may be incorrect. Do not use "
    "this app to make healthcare decisions."
)


# =========================================================
# HOME
# =========================================================
if page == "Home":
    st.header("Project Overview")

    st.markdown(
        """
        This interface demonstrates a research prototype combining a
        convolutional neural network (CNN) backbone with Transformer
        components for three-class chest X-ray classification.

        **Supported output classes**
        - Normal
        - Tuberculosis
        - COVID-19

        **Available research functions**
        - Image classification and class scores
        - Grad-CAM visualization, when gradients are connected
        - Experimental uncertainty and review flags
        - Synthetic patient-history score comparison
        - PDF report generation and download
        - Viewing and downloading existing project CSV results
        """
    )

    st.subheader("Model information")
    st.write(f"Model file: `{MODEL_PATH.name}`")
    st.write(f"Model file exists: **{MODEL_PATH.exists()}**")
    st.write("Input image size: 224 × 224 pixels")
    st.write("Number of output classes: 3")

    st.info(
        "The patient-history comparison section uses synthetic records "
        "for demonstration. Score changes do not establish disease "
        "progression or treatment response."
    )


# =========================================================
# CLASSIFICATION
# =========================================================
elif page == "X-ray Classification":
    st.header("Upload and Classify a Chest X-ray")

    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["png", "jpg", "jpeg"],
        key="classification_upload",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            left, right = st.columns(2)

            with left:
                st.subheader("Uploaded image")
                st.image(image, use_container_width=True)

            with right:
                st.subheader("Classification")

                if st.button(
                    "Run classification",
                    type="primary",
                    key="run_classification",
                ):
                    with st.spinner("Loading model and predicting..."):
                        model = load_trained_model()
                        input_batch, display_image = prepare_image(image)

                        scores = model.predict(input_batch, verbose=0)[0]
                        scores = np.asarray(scores, dtype=np.float64)

                        if (
                            len(scores) != len(CLASS_NAMES)
                            or not np.all(np.isfinite(scores))
                        ):
                            raise ValueError(
                                "Model returned unexpected or invalid scores."
                            )

                        predicted_index = int(np.argmax(scores))
                        predicted_class = CLASS_NAMES[predicted_index]
                        uncertainty = calculate_uncertainty(scores)

                        st.session_state.prediction_result = {
                            "scores": scores.tolist(),
                            "predicted_index": predicted_index,
                            "predicted_class": predicted_class,
                        }
                        st.session_state.uncertainty_result = uncertainty
                        st.session_state.original_image_bytes = (
                            uploaded_file.getvalue()
                        )
                        st.session_state.image_filename = uploaded_file.name

                        # Invalidate a previous image's Grad-CAM.
                        st.session_state.gradcam_png = None
                        st.session_state.gradcam_error = None

                        st.session_state.display_image = display_image
                        st.session_state.input_batch = input_batch.numpy()

            result = st.session_state.prediction_result

            if (
                result is not None
                and st.session_state.image_filename == uploaded_file.name
            ):
                st.divider()
                st.subheader("Prediction results")

                st.metric("Predicted class", result["predicted_class"])

                score_columns = st.columns(3)
                for column, class_name, score in zip(
                    score_columns, CLASS_NAMES, result["scores"]
                ):
                    with column:
                        st.metric(class_name, f"{score * 100:.2f}%")
                        st.progress(float(np.clip(score, 0, 1)))

                st.caption(
                    "Scores are model outputs and may not be calibrated "
                    "medical probabilities."
                )

                uncertainty = st.session_state.uncertainty_result
                if uncertainty["review_flag"]:
                    st.warning(
                        "Experimental review flag: this image meets the "
                        "project's uncertainty rule. This is not a medical "
                        "triage decision."
                    )
                else:
                    st.info(
                        "Experimental review flag: not triggered by the "
                        "current project rule. This does not confirm that "
                        "the prediction is correct."
                    )

        except Exception as exc:
            st.error("Classification failed.")
            st.exception(exc)


# =========================================================
# GRAD-CAM
# =========================================================
elif page == "Grad-CAM Explainability":
    st.header("Grad-CAM Explainability")

    result = st.session_state.prediction_result

    if result is None:
        st.info(
            "First classify an image on the X-ray Classification page."
        )
    else:
        st.write(f"Current prediction: **{result['predicted_class']}**")

        if st.button("Generate Grad-CAM", type="primary"):
            with st.spinner("Calculating Grad-CAM..."):
                model = load_trained_model()
                input_batch = tf.convert_to_tensor(
                    st.session_state.input_batch,
                    dtype=tf.float32,
                )

                _, overlay_png, error_message = generate_gradcam(
                    model,
                    input_batch,
                    st.session_state.display_image,
                )

                st.session_state.gradcam_png = overlay_png
                st.session_state.gradcam_error = error_message

        if st.session_state.gradcam_png is not None:
            st.image(
                st.session_state.gradcam_png,
                caption="Grad-CAM overlay — exploratory visualization",
                use_container_width=True,
            )

            st.download_button(
                "Download Grad-CAM overlay (PNG)",
                data=st.session_state.gradcam_png,
                file_name="gradcam_overlay.png",
                mime="image/png",
            )

            st.caption(
                "Highlighted regions are not guaranteed to correspond "
                "to disease-specific anatomy or medically meaningful evidence."
            )

        elif st.session_state.gradcam_error:
            st.error(st.session_state.gradcam_error)
            st.info(
                "No heatmap was displayed because the Grad-CAM computation "
                "did not produce a valid connected result."
            )
        else:
            st.info("Click **Generate Grad-CAM** to attempt the visualization.")


# =========================================================
# UNCERTAINTY ANALYSIS
# =========================================================
elif page == "Uncertainty Analysis":
    st.header("Uncertainty Analysis")

    result = st.session_state.prediction_result
    uncertainty = st.session_state.uncertainty_result

    if result is not None and uncertainty is not None:
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Maximum model score",
            f"{uncertainty['max_score']:.4f}",
        )
        c2.metric(
            "Normalized entropy",
            f"{uncertainty['normalized_entropy']:.4f}",
        )
        c3.metric(
            "Top-two score margin",
            f"{uncertainty['margin']:.4f}",
        )

        st.write("Class scores")
        score_df = pd.DataFrame(
            {
                "Class": CLASS_NAMES,
                "Model score": result["scores"],
            }
        )
        st.bar_chart(score_df.set_index("Class"))

        if uncertainty["review_flag"]:
            st.warning(
                "REVIEW FLAGGED by the exploratory rule: maximum score "
                "< 0.80 OR normalized entropy > 0.40."
            )
        else:
            st.success(
                "NOT FLAGGED by the exploratory rule. This does not mean "
                "the model is correct or the image is medically normal."
            )

        st.caption(
            "Entropy and score margin describe model output uncertainty; "
            "they do not establish clinical risk."
        )
    else:
        st.info(
            "Classify an image first to calculate its uncertainty indicators."
        )

    st.divider()
    st.subheader("Existing test uncertainty results")
    show_csv_preview(
        "Test uncertainty and review results",
        UNCERTAINTY_CSV,
    )
    show_csv_preview(
        "Review flag summary",
        REVIEW_SUMMARY_CSV,
    )


# =========================================================
# PATIENT HISTORY COMPARISON
# =========================================================
elif page == "Patient History Comparison":
    st.header("Patient History Comparison")

    st.warning(
        "Demonstration feature: use synthetic records only. Do not enter "
        "identifying or sensitive real patient information. Score changes "
        "do not establish disease progression."
    )

    history_df = load_csv_if_available(HISTORY_CSV)

    if history_df is not None:
        st.subheader("Existing synthetic comparison CSV")
        st.caption(f"Loaded: {HISTORY_CSV.name}")
        st.dataframe(history_df, use_container_width=True)

        st.download_button(
            "Download synthetic comparison CSV",
            data=HISTORY_CSV.read_bytes(),
            file_name=HISTORY_CSV.name,
            mime="text/csv",
        )
    else:
        st.info(
            "The synthetic comparison CSV was not found. "
            "You can still use the manual score comparison below."
        )

    result = st.session_state.prediction_result

    if result is None:
        st.info(
            "Classify an image first to compare its current model scores."
        )
    else:
        st.subheader("Compare with previous model scores")

        st.write(
            "Enter example scores from a previous model output. "
            "The three values will be normalized to sum to 1."
        )

        with st.form("history_comparison_form"):
            previous_normal = st.number_input(
                "Previous Normal score",
                min_value=0.0,
                max_value=1.0,
                value=0.80,
                step=0.01,
            )
            previous_tb = st.number_input(
                "Previous Tuberculosis score",
                min_value=0.0,
                max_value=1.0,
                value=0.10,
                step=0.01,
            )
            previous_covid = st.number_input(
                "Previous COVID-19 score",
                min_value=0.0,
                max_value=1.0,
                value=0.10,
                step=0.01,
            )

            compare_clicked = st.form_submit_button(
                "Compare model outputs"
            )

        if compare_clicked:
            previous_scores = np.array(
                [previous_normal, previous_tb, previous_covid],
                dtype=float,
            )

            if previous_scores.sum() <= 0:
                st.error("At least one previous score must be greater than zero.")
            else:
                previous_scores = previous_scores / previous_scores.sum()
                current_scores = np.asarray(result["scores"], dtype=float)
                difference = current_scores - previous_scores

                comparison_df = pd.DataFrame(
                    {
                        "Class": CLASS_NAMES,
                        "Previous score": previous_scores,
                        "Current score": current_scores,
                        "Change (current - previous)": difference,
                    }
                )

                st.dataframe(
                    comparison_df.style.format(
                        {
                            "Previous score": "{:.4f}",
                            "Current score": "{:.4f}",
                            "Change (current - previous)": "{:+.4f}",
                        }
                    ),
                    use_container_width=True,
                )

                st.bar_chart(
                    comparison_df.set_index("Class")[
                        ["Previous score", "Current score"]
                    ]
                )

                st.caption(
                    "This is a numerical comparison of model outputs only. "
                    "It is not evidence of disease progression, improvement, "
                    "or treatment response."
                )


# =========================================================
# RESEARCH REPORTS
# =========================================================
elif page == "Research Reports":
    st.header("Generate and Download a PDF Report")

    result = st.session_state.prediction_result
    uncertainty = st.session_state.uncertainty_result

    if result is None or uncertainty is None:
        st.info(
            "Classify an image first to generate a report for that result."
        )
    else:
        st.write(f"Prediction: **{result['predicted_class']}**")
        st.write(f"Image: `{st.session_state.image_filename}`")

        if st.button("Generate PDF report", type="primary"):
            try:
                pdf_bytes = create_pdf_report(
                    predicted_class=result["predicted_class"],
                    scores=result["scores"],
                    uncertainty=uncertainty,
                    image_name=st.session_state.image_filename or "uploaded_image",
                    gradcam_png=st.session_state.gradcam_png,
                )

                st.session_state.pdf_bytes = pdf_bytes
                st.success("PDF report generated.")

            except Exception as exc:
                st.error("Could not generate the PDF report.")
                st.exception(exc)

        if st.session_state.get("pdf_bytes"):
            st.download_button(
                "Download PDF report",
                data=st.session_state.pdf_bytes,
                file_name="chest_xray_research_report.pdf",
                mime="application/pdf",
            )

    st.divider()
    st.subheader("Existing report files")

    for report_path in [REPORT_TXT, REPORT_JSON]:
        if report_path.exists():
            st.write(f"**{report_path.name}**")
            st.download_button(
                f"Download {report_path.name}",
                data=report_path.read_bytes(),
                file_name=report_path.name,
                mime=(
                    "application/json"
                    if report_path.suffix == ".json"
                    else "text/plain"
                ),
                key=f"download_{report_path.name}",
            )
        else:
            st.caption(f"Not found: {report_path.relative_to(PROJECT_DIR)}")


# =========================================================
# EXISTING PROJECT RESULTS
# =========================================================
elif page == "Project Results":
    st.header("Existing Project Results")

    st.markdown(
        "Browse and download available CSV files from the research project."
    )

    show_csv_preview(
        "Evaluation error results",
        EVALUATION_CSV,
    )
    show_csv_preview(
        "Test uncertainty and review results",
        UNCERTAINTY_CSV,
    )
    show_csv_preview(
        "Review flag summary",
        REVIEW_SUMMARY_CSV,
    )
    show_csv_preview(
        "Synthetic patient-history comparison",
        HISTORY_CSV,
    )

    st.divider()
    st.subheader("Learning-rate comparison")

    learning_rate_csv = (
        PROJECT_DIR / "hyperparameter_tuning" / "learning_rate_comparison.csv"
    )
    show_csv_preview(
        "Learning-rate comparison results",
        learning_rate_csv,
    )

    learning_rate_results_csv = (
        PROJECT_DIR / "hyperparameter_tuning" / "learning_rate_results.csv"
    )
    show_csv_preview(
        "Learning-rate experiment results",
        learning_rate_results_csv,
    )


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Academic research prototype • Hybrid CNN–Transformer • "
    "Not validated for clinical use • Not a medical diagnostic tool"
)
