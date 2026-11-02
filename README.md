# Hybrid CNN-Transformer for Chest X-Ray Classification

A research prototype for classifying chest X-ray images into **Normal, Tuberculosis (TB), and COVID-19** categories using a hybrid Convolutional Neural Network (CNN) and Transformer architecture.

> **Disclaimer:** This project is intended for research and educational purposes only. It is not a medical device and must not be used to diagnose disease or guide clinical decisions.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Performance](#performance)
- [Explainability and Uncertainty](#explainability-and-uncertainty)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

## Overview

Chest X-ray classification is a computer vision task in which deep learning models learn visual patterns from medical images. This project explores a hybrid architecture combining CNN-based feature extraction with Transformer-based feature processing.

The prototype includes model evaluation, Grad-CAM visualizations, uncertainty analysis, and a Streamlit interface for research demonstrations.

## Key Features

- Three-class chest X-ray classification.
- MobileNetV2-based feature extraction.
- Transformer-based feature processing.
- Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.
- Grad-CAM visualizations for exploring image regions that influence predictions.
- Prediction confidence and uncertainty analysis.
- Experimental patient-history comparison using synthetic records.
- Streamlit web interface for interactive demonstrations.

## Model Architecture

The model combines:

1. **CNN feature extractor:** A pretrained MobileNetV2 backbone extracts visual features from input images.
2. **Transformer component:** Transformer-based processing models relationships between extracted features.
3. **Classification head:** Produces scores for three classes.

### Input and output

- Input image size: `224 × 224` pixels.
- Number of output classes: `3`.
- Class mapping:
  - `0` — Normal
  - `1` — Tuberculosis (TB)
  - `2` — COVID-19

## Dataset

The experiment uses three dataset splits:

| Split | Images |
|---|---:|
| Training | 12,605 |
| Validation | 2,701 |
| Testing | 2,702 |
| **Total** | **18,008** |

The reported class distribution is:

| Class | Training | Validation | Testing |
|---|---:|---:|---:|
| Normal | 9,584 | 2,054 | 2,054 |
| TB | 490 | 105 | 105 |
| COVID-19 | 2,531 | 542 | 543 |

The dataset is imbalanced, particularly for the TB class. This is important when interpreting model performance.

**Dataset access:** Dataset files are not included in this repository. Use only datasets you have permission to access and use, and follow their applicable licenses and terms.

## Performance

The recorded test accuracy for the evaluated three-class model was **96.52%** on 2,702 test images.

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Normal | 97.76% | 97.91% | 97.84% |
| TB | 93.52% | 96.19% | 94.84% |
| COVID-19 | 92.36% | 91.34% | 91.85% |

These are results from the reported experiment, not a guarantee of performance on new images or other populations.

## Explainability and Uncertainty

### Grad-CAM

Grad-CAM visualizations are included to help explore which image regions influence model predictions. These heatmaps are exploratory and do not establish that the model is attending to medically meaningful features.

### Uncertainty analysis

The project explores prediction confidence, entropy, and the margin between the top two class probabilities. An experimental review flag identifies selected predictions for further inspection.

This flag is a research mechanism, not a clinically validated safety system. Model confidence should not be interpreted as a probability that a patient has a disease.

### Synthetic patient-history comparison

The project includes an experimental comparison using synthetic patient-history records. It does not establish real-world disease progression or clinical outcomes.

## Installation

### Requirements

- Python 3.10 or another Python version compatible with the dependencies in `requirements.txt`.
- Git.
- The project source code.
- Access to the dataset required by the notebook.

### 1. Clone the repository

```bash
git clone https://github.com/indhuprasanna/hybrid-cnn-transformer-chest-xray.git
cd hybrid-cnn-transformer-chest-xray
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application from the project root:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

The application provides research-oriented pages for X-ray classification, Grad-CAM visualization, uncertainty analysis, patient-history comparison, and research reports.

**Model availability:** Trained model checkpoints are excluded from the repository. Before using the classification or explainability pages, check the application code for the expected model path and follow the project's documented model-loading process.

## Project Structure

```text
hybrid-cnn-transformer-chest-xray/
├── ai_reports/
├── evaluation_results/
├── patient_history_comparison/
├── uncertainty_analysis/
├── app.py
├── train_model.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

The exact generated files and available features may vary as the project develops.

## Limitations

- The dataset is class-imbalanced.
- Reported metrics reflect a specific experimental test split.
- Independent external validation has not been established here.
- Grad-CAM visualizations do not prove clinical reasoning.
- Confidence and uncertainty measures are not clinically validated.
- Synthetic patient-history comparisons do not demonstrate actual patient outcomes.
- Dataset provenance, licensing, image quality, and potential data leakage must be considered when interpreting results.

**This model must not be used for diagnosis, screening, treatment decisions, or other clinical purposes.**

## Future Improvements

- Evaluate performance on independent external datasets.
- Investigate class imbalance and calibration.
- Add reproducible training and evaluation scripts.
- Document dataset provenance and preprocessing in greater detail.
- Improve model interpretability and evaluate explanations systematically.
- Add automated tests and clearer model setup instructions.



## License

No license has been specified yet. Unless a license is added, others generally do not receive permission to reuse, distribute, or modify this project's code beyond applicable legal exceptions. Add an appropriate license only after deciding how the project should be shared.