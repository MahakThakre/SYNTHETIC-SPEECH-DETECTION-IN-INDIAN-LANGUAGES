# Synthetic Speech Detection in Indian Languages

Detect AI-generated speech ("deepfakes") in 13 Indian languages using a state-of-the-art deep learning pipeline.

## Overview

This project leverages deep learning to identify synthetic (AI-generated) speech across multiple Indian languages. It uses advanced feature extraction and preprocessing techniques to achieve high reliability in detecting deepfake audio.

- **Model:** DistilHuBERT (Hugging Face)
- **Frameworks:** PyTorch, Hugging Face Transformers
- **Languages Supported:** 13 Indian languages
- **Dataset:** 33,737 speech samples

## Key Features

- Detects synthetic speech with high reliability (ROC-AUC: 0.9735, >95% accuracy)
- Utilizes advanced preprocessing and feature extraction techniques (5+ applied)
- GPU-accelerated with CUDA support for fast training and inference

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Tqdm
- NumPy
- Data Collator
- AutoFeatureExtractor (DistilHuBERT)
- GPU (CUDA)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MahakThakre/SYNTHETIC-SPEECH-DETECTION-IN-INDIAN-LANGUAGES.git
   cd SYNTHETIC-SPEECH-DETECTION-IN-INDIAN-LANGUAGES
   ```

2. Enable GPU support:
   - Make sure CUDA is installed and available.

---
