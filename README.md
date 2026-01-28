# Automated Food Freshness Detection Application (Group 12)

## Project Overview
This project applies deep learning to classify food items as "Fresh" or "Rotten." It is designed to be a user-friendly tool for reducing food waste by providing instant freshness assessments with confidence levels.

## Technical Components & Roles
- **Galoh Safiya**: Data Architecture, Automated Preprocessing (Capping/Cleaning), and MobileNetV2 Training with Weighted Sampling.
- **Marsha Binti Lana**: EfficientNet Implementation and Hyperparameter Comparison.
- **Nur Arissa Hanani**: UI/UX Design (Streamlit), Application Flow, and Quantitative Analysis.

## Hardware Support
The codebase is platform-agnostic and automatically detects the best available hardware:
- **Apple Silicon (M2/M3)**: Utilizes `MPS` (Metal Performance Shaders).
- **NVIDIA GPU**: Utilizes `CUDA`.
- **Intel Mac/Standard Windows**: Defaults to `CPU`.

## Setup & Execution

### 1. Installation
```bash
pip install -r requirements.txt