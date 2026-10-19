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

- Input image size