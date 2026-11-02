# PulmoScan AI: Hybrid CNN-Vision Transformer for Chest Radiography Diagnosis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end medical AI diagnostic pipeline integrating the localized inductive bias of **MobileNetV2** with the long-range global contextual reasoning of a **Multi-Head Self-Attention (MHSA) Vision Transformer**. The system evaluates frontal chest radiographs across **COVID-19**, **Normal**, and **Tuberculosis** conditions, coupling diagnostic predictions with **Monte Carlo Dropout Epistemic Uncertainty Estimation**, **Grad-CAM visual localization**, and **longitudinal disease trajectory tracking**.

---

## Key Highlights

- **Hybrid Inductive Architecture**: Bridges fine-grained local feature extraction (edges, cavitations, opacities) with global spatial dependencies across both lung fields via 8-head self-attention.
- **Safety Triage via Uncertainty**: Employs Monte Carlo Dropout ($T = 10$ stochastic forward passes) to compute epistemic variance ($\sigma^2$). Ambiguous cases ($\sigma^2 \ge 0.05$ or confidence $< 80\%$) trigger an automated referral to a human radiologist.
- **Explainability & Localization**: Generates real-time Grad-CAM attention heatmaps overlaid on anatomical lung regions for clinical verification.
- **Longitudinal Monitoring**: Compares sequential chest radiographs over time to index disease trajectory as *Improving*, *Stable*, or *Worsening*.
- **Lightweight Clinical Deployment**: Compact footprint of **14.52 MB** and rapid inference time (~81 ms/epoch), designed for edge clinical workstations and low-resource settings.

---

## 5-Model Benchmark & Ablation Study

Evaluated on a strictly balanced cohort of **2,100 frontal chest radiographs** (700 COVID-19, 700 Normal, 700 Tuberculosis):

| Architecture | Paradigm | Validation Accuracy (%) | Holdout Test Accuracy (%) | Macro F1-Score | Tuberculosis Sensitivity | Model Weight Size | Inference / Training Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DenseNet-121** | Baseline Deep CNN | 78.10% | 78.10% | 0.7792 | 82.50% | 27.76 MB | ~205 s/epoch |
| **Swin-T** | Hierarchical ViT | 79.68% | 79.68% | 0.7954 | 84.00% | 107.80 MB | ~262 s/epoch |
| **Transformer Encoder** | Standalone ViT | 74.29% | 75.16% | 0.7533 | 77.36% | 18.60 MB | ~94 s/epoch |
| **MobileNetV2** | Standalone CNN | 95.24% | 95.91% | 0.9590 | 100.00% | 9.10 MB | ~119 s/epoch |
| **Proposed Hybrid (Ours)** | **CNN + MHSA ViT** | **91.75%** | **94.03%** | **0.9401** | **100.00%** | **14.52 MB** | **~81 s/epoch** |

---

## System Architecture

```text
[Input Chest Radiograph]
        │
        ▼
[Preprocessing & Augmentation] ──► (224x224 resize, ImageNet normalization, random flips/rotations)
        │
        ▼
[MobileNetV2 CNN Backbone]     ──► (Extracts local spatial feature maps: 7x7x1280)
        │
        ▼
[1x1 Conv & Token Projection]  ──► (Channel reduction to d_model=256; unrolls to 49 visual tokens)
        │
        ▼
[Transformer Encoder (MHSA)]   ──► (2 layers, 8 attention heads; captures bilateral lung context)
        │
        ▼
[Monte Carlo Dropout (T=10)]   ──► Computes epistemic uncertainty (variance σ²)
        │
        ├──► Path A: Confident Case (σ² < 0.05, Conf ≥ 80%) ──► Automated Diagnosis (COVID/Normal/TB)
        └──► Path B: Ambiguous Case (σ² ≥ 0.05)            ──► Flagged for Radiologist Review
        │
        ▼
[Clinical Interpretability]    ──► Grad-CAM Heatmap Localization & Longitudinal Trajectory Tracking

---

## Project Contributors

- **Indhu Prasanna** - [@indhuprasanna](https://github.com/indhuprasanna)
- **Jahnavi Gummadi**
- **Kadiri Lalasa**