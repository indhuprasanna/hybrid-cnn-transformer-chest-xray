import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
from fpdf import FPDF

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="PulmoScan AI - Clinical CAD & Triage",
    page_icon="🫁",
    layout="wide"
)

# ----------------- SYSTEM SETUP -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['COVID19', 'Normal', 'Tuberculosis']

# ----------------- MODEL ARCHITECTURE -----------------
class HybridCXRModel(nn.Module):
    def __init__(self, num_classes=3, d_model=256, nhead=8, num_layers=2, dropout_p=0.3):
        super(HybridCXRModel, self).__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        self.features = mobilenet.features
        self.conv_proj = nn.Conv2d(in_channels=1280, out_channels=d_model, kernel_size=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout_p, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.conv_proj(feat)
        tokens = feat.flatten(2).permute(0, 2, 1) + self.pos_embedding
        attended = self.transformer_encoder(tokens)
        pooled = attended.mean(dim=1)
        return self.classifier(self.dropout(pooled))

@st.cache_resource
def load_trained_model():
    model = HybridCXRModel(num_classes=3).to(device)
    model_path = "best_hybrid_mobilenetv2_transformer.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

model = load_trained_model()

# Preprocessing transforms (Identical to Test Phase)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------- IMAGE SANITIZATION -----------------
def sanitize_cxr_image(uploaded_file):
    """Converts uploaded file into standard 3-channel RGB image."""
    img = Image.open(uploaded_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# ----------------- CLEAN GRAD-CAM (NO RESIDUAL STATE) -----------------
def generate_gradcam_clean(model, input_tensor, class_idx):
    target_layer = model.features[-1]
    gradients = []
    activations = []

    def f_hook(module, inp, out):
        activations.append(out)

    def b_hook(module, g_in, g_out):
        gradients.append(g_out[0])

    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_full_backward_hook(b_hook)

    model.eval()
    output = model(input_tensor)
    score = output[0, class_idx]

    model.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    grad = gradients[0].detach().cpu().numpy()[0]
    act = activations[0].detach().cpu().numpy()[0]
    model.zero_grad()

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    return cam

# ----------------- DETERMINISTIC & SAFE INFERENCE -----------------
def run_calibrated_inference(model, input_tensor, t_passes=10):
    # Always ensure model is in eval mode so BatchNorm is NEVER altered
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx] * 100)

    # Isolated MC Dropout: Enable ONLY nn.Dropout modules, keeping BatchNorm in eval!
    mc_probs = []
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    torch.manual_seed(42)
    with torch.no_grad():
        for _ in range(t_passes):
            mc_logits = model(input_tensor)
            mc_probs.append(F.softmax(mc_logits, dim=1).detach().cpu().numpy()[0])

    # Return model to full eval
    model.eval()

    mc_probs = np.array(mc_probs)
    variance = np.var(mc_probs, axis=0)
    uncertainty_var = float(variance[pred_idx])

    # Referral & Risk Logic
    if confidence >= 80.0 and uncertainty_var <= 0.05:
        decision = "AUTOMATED PREDICT (High Reliability)"
        risk_level = "High Pathological Severity" if pred_class != "Normal" else "Normal / Low Risk"
    else:
        decision = "REFER TO RADIOLOGIST (Review Required)"
        risk_level = "Indeterminate / Requires Secondary Scan"

    return pred_class, confidence, uncertainty_var, decision, risk_level, pred_idx, probs

# ----------------- PDF EXPORT BUILDER -----------------
def generate_pdf_report(patient_id, pred_class, conf, uncert, decision, risk, obs, orig_img, overlay_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "PULMOSCAN AI - DIAGNOSTIC REPORT", ln=True, align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Patient Identifier: {patient_id}", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, f"Predicted Diagnosis: {pred_class}", ln=True)
    pdf.cell(0, 7, f"Diagnostic Confidence: {conf:.2f}%", ln=True)
    pdf.cell(0, 7, f"Model Uncertainty Variance: {uncert:.5f}", ln=True)
    pdf.cell(0, 7, f"Clinical Triage Action: {decision}", ln=True)
    pdf.cell(0, 7, f"Risk Stratification: {risk}", ln=True)
    pdf.multi_cell(0, 7, f"Clinical Observations: {obs}")
    pdf.ln(6)

    orig_path = "temp_orig.png"
    overlay_path = "temp_overlay.png"
    orig_img.save(orig_path)
    Image.fromarray(overlay_img).save(overlay_path)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(95, 8, "Input Chest Radiograph", align="C")
    pdf.cell(95, 8, "Grad-CAM Activation Map", ln=True, align="C")
    pdf.image(orig_path, x=20, y=pdf.get_y(), w=75, h=75)
    pdf.image(overlay_path, x=115, y=pdf.get_y(), w=75, h=75)

    pdf_buffer = io.BytesIO()
    pdf_bytes = pdf.output(dest='S')
    pdf_buffer.write(pdf_bytes)
    pdf_buffer.seek(0)
    return pdf_buffer

# ----------------- STREAMLIT UI -----------------
st.title("🫁 PulmoScan AI: Explainable CXR Triage & Trajectory System")
st.markdown("Multi-Class Pathology Detection (**Normal / Tuberculosis / COVID-19**) with Grad-CAM & Uncertainty-Driven Referral.")

tab1, tab2 = st.tabs(["Diagnostic Triage & Reporting", "Longitudinal Patient History Tracking"])

# ================= TAB 1: Diagnostic Mode =================
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Patient Details")
        patient_id = st.text_input("Patient ID", value="PID-98241")
        uploaded_file = st.file_uploader("Upload Chest Radiograph (PA/AP)", type=["png", "jpg", "jpeg"], key="single_scan")
        ai_obs = st.text_area("Clinical Notes", value="Routine radiographic evaluation for pulmonary respiratory symptoms.")

    with col2:
        if uploaded_file is not None:
            orig_pil = sanitize_cxr_image(uploaded_file)
            input_tensor = val_transforms(orig_pil).unsqueeze(0).to(device)

            pred_class, conf, uncert, decision, risk, pred_idx, probs = run_calibrated_inference(model, input_tensor)
            cam = generate_gradcam_clean(model, input_tensor, pred_idx)

            img_np = np.array(orig_pil.resize((224, 224)))
            cam_color = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_np, 0.65, cam_color, 0.35, 0)

            st.subheader("Visual Diagnosis & Localization")
            sub1, sub2 = st.columns(2)
            with sub1:
                st.image(orig_pil, caption="Sanitized Chest X-Ray", use_container_width=True)
            with sub2:
                st.image(overlay, caption=f"Grad-CAM: Focal Focus for {pred_class}", use_container_width=True)

            # Metrics Card
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Condition", pred_class)
            m2.metric("Confidence Score", f"{conf:.2f}%")
            m3.metric("Uncertainty Variance", f"{uncert:.5f}")

            # Probability Breakdown
            st.markdown("##### Class Probability Distribution")
            p1, p2, p3 = st.columns(3)
            p1.progress(float(probs[0]), text=f"COVID-19: {probs[0]*100:.1f}%")
            p2.progress(float(probs[1]), text=f"Normal: {probs[1]*100:.1f}%")
            p3.progress(float(probs[2]), text=f"Tuberculosis: {probs[2]*100:.1f}%")

            # Referral Status
            if "AUTOMATED" in decision:
                st.success(f"**Triage Action:** {decision} | **Risk Stratification:** {risk}")
            else:
                st.warning(f"**Triage Action:** {decision} | **Risk Stratification:** {risk}")

            # PDF Download
            pdf_data = generate_pdf_report(patient_id, pred_class, conf, uncert, decision, risk, ai_obs, orig_pil, overlay)
            st.download_button(
                label="📄 Download Diagnostic PDF Report",
                data=pdf_data,
                file_name=f"PulmoScan_Report_{patient_id}.pdf",
                mime="application/pdf"
            )

# ================= TAB 2: Longitudinal History Mode =================
with tab2:
    st.subheader("Longitudinal Disease Trend & Patient Trajectory Tracking")
    st.write("Upload sequential chest radiographs to track disease recovery or deterioration over time.")

    c1, c2 = st.columns(2)
    with c1:
        scan_1 = st.file_uploader("Upload Scan 1 (Baseline)", type=["png", "jpg", "jpeg"], key="hist_scan1")
        date_1 = st.date_input("Scan 1 Date", key="d1")
    with c2:
        scan_2 = st.file_uploader("Upload Scan 2 (Follow-up)", type=["png", "jpg", "jpeg"], key="hist_scan2")
        date_2 = st.date_input("Scan 2 Date", key="d2")

    if scan_1 and scan_2:
        img1 = sanitize_cxr_image(scan_1)
        img2 = sanitize_cxr_image(scan_2)

        t1 = val_transforms(img1).unsqueeze(0).to(device)
        t2 = val_transforms(img2).unsqueeze(0).to(device)

        cls1, conf1, _, _, _, _, probs1 = run_calibrated_inference(model, t1)
        cls2, conf2, _, _, _, _, probs2 = run_calibrated_inference(model, t2)

        # Risk scoring
        risk_score_1 = (1.0 - probs1[1]) * 100.0
        risk_score_2 = (1.0 - probs2[1]) * 100.0

        diff = risk_score_2 - risk_score_1
        if diff < -8.0:
            trend = "Improving"[cite: 1]
            alert_ui = st.success
        elif abs(diff) <= 8.0:
            trend = "Stable"[cite: 1]
            alert_ui = st.info
        else:
            trend = "Worsening"[cite: 1]
            alert_ui = st.error

        res1, res2 = st.columns(2)
        with res1:
            st.markdown(f"**Baseline Scan ({date_1})**")
            st.write(f"Diagnosis: **{cls1}** | Pathological Risk: **{risk_score_1:.1f}%**")
            st.image(img1, width=280)
        with res2:
            st.markdown(f"**Follow-up Scan ({date_2})**")
            st.write(f"Diagnosis: **{cls2}** | Pathological Risk: **{risk_score_2:.1f}%**")
            st.image(img2, width=280)

        alert_ui(f"### Patient Recovery Trajectory: **{trend}**")[cite: 1]

        # Horizontal progression chart matching your notebook design
        fig_trend, ax = plt.subplots(figsize=(8, 2.4))
        dates = [f"Baseline ({date_1})", f"Follow-up ({date_2})"]
        scores = [risk_score_1, risk_score_2]
        bar_colors = ['#2b5c8f', '#27ae60' if trend == 'Improving' else ('#e67e22' if trend == 'Stable' else '#c0392b')]
        ax.barh(dates, scores, color=bar_colors, height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Pulmonary Pathological Risk (%)")
        ax.set_title("Disease Trajectory Over Time")
        for i, v in enumerate(scores):
            ax.text(v + 1.5, i, f"{v:.1f}%", va='center', fontweight='bold')
        st.pyplot(fig_trend)