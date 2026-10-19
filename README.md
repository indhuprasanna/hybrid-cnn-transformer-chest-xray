# Hybrid Deep Learning Framework for Pulmonary Tuberculosis Detection from Chest X-ray Images

An AI-powered deep learning framework for automated Pulmonary Tuberculosis (TB) detection using chest X-ray images. This project combines the feature extraction capability of **MobileNetV2** with the contextual learning ability of a **Transformer Encoder** through feature fusion, enabling accurate and efficient TB diagnosis.

---

## Overview

Tuberculosis (TB) remains one of the world's leading infectious diseases, particularly in regions with limited access to expert radiologists. Early detection is essential for timely treatment and reducing disease transmission.

This project presents a hybrid deep learning framework that combines Convolutional Neural Networks (CNNs) and Transformer-based attention mechanisms to classify chest X-ray images as **Tuberculosis Positive** or **Normal**, assisting healthcare professionals in rapid and reliable diagnosis.

---

## Features

- Hybrid CNN + Transformer architecture
- MobileNetV2 for efficient feature extraction
- Transformer Encoder for global contextual learning
- Feature Fusion for improved classification
- Automated Pulmonary TB detection
- High classification accuracy (96%)
- Simple web interface using Flask
- Lightweight model suitable for deployment

---

## Model Architecture

```
Chest X-ray Image
        │
        ▼
Image Preprocessing
        │
        ▼
MobileNetV2
(Local Feature Extraction)
        │
        ▼
Transformer Encoder
(Global Context Learning)
        │
        ▼
Feature Fusion
        │
        ▼
Dense Classification Layer
        │
        ▼
TB / Normal Prediction
```

---

## Dataset

- Chest X-ray Images
- Total Images: **4,200**
- Binary Classification:
  - Tuberculosis
  - Normal

The dataset was preprocessed using:

- Image resizing
- Normalization
- Data augmentation

---

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- Transformer Encoder
- Flask
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn

---

## Repository Structure

```
Pulmonary-TB-Detection/
│
├── .gitignore                 # Git ignored files
├── README.md                  # Project documentation
├── app.py                     # Flask web application
├── requirements.txt           # Project dependencies
├── tb_detection_model.h5      # Trained hybrid deep learning model
└── train_model.ipynb          # Model training notebook
```

---

## Workflow

1. Load Chest X-ray dataset
2. Preprocess and augment images
3. Train the hybrid MobileNetV2 + Transformer model
4. Save the trained model (.h5)
5. Launch the Flask application
6. Upload a Chest X-ray image
7. Predict whether the patient is **TB Positive** or **Normal**

---

## Performance

| Metric | Value |
|---------|--------|
| Model | Hybrid MobileNetV2 + Transformer |
| Accuracy | **96%** |
| Dataset | 4,200 Chest X-ray Images |
| Classification | Binary |

---

## Results

The hybrid architecture effectively combines the strengths of CNNs and Transformers, resulting in:

- 96% Classification Accuracy
- Improved contextual feature learning
- Efficient feature extraction
- Better generalization through feature fusion

---

## Future Enhancements
- Explainable AI using Grad-CAM
- Multi-class lung disease classification
- Clinical report generation
- Cloud deployment
- Mobile application integration

---

## Applications

- AI-assisted Tuberculosis screening
- Clinical decision support
- Rural healthcare diagnostics
- Automated chest X-ray analysis
- Public health screening programs

---

## Authors
**Kanakadandi Vidya Sai Prasanna**, 
**Kadiri Lalasa**, 
**Jahnavi Gummadi**.
Bachelor of Engineering (Computer Science and Engineering)

---

## License
This project is intended for academic and research purposes.
