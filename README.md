# SignApp

SignApp is a static American Sign Language (ASL) alphabet recognition system that extracts hand landmarks using MediaPipe and classifies signs with a TensorFlow neural network.

## Overview

The project processes ASL alphabet images, extracts normalized hand landmarks, engineers additional geometric features, and uses a neural-network classifier to recognize 29 sign classes.

## Pipeline

1. Load ASL alphabet images
2. Detect hand landmarks using MediaPipe Hands
3. Extract 21 hand landmarks with x, y, and z coordinates
4. Normalize landmarks using wrist-centered coordinates and scale normalization
5. Mirror left-hand landmarks into a common orientation
6. Add geometric pinch-distance features
7. Train a TensorFlow/Keras neural-network classifier
8. Evaluate using a held-out test set and confusion matrix

## Features

- MediaPipe-based hand landmark extraction
- Wrist-centered and scale-normalized landmarks
- Left-hand orientation normalization
- Geometric feature engineering
- Gaussian jitter augmentation
- Batch normalization and dropout
- Cosine-decay learning-rate scheduling
- Label smoothing
- Early stopping and best-model checkpointing
- Training-curve and confusion-matrix visualization

## Model

The classifier uses a fully connected neural network:

```text
65 Input Features
       ↓
Dense 1024 + BatchNorm + ReLU + Dropout
       ↓
Dense 512 + BatchNorm + ReLU + Dropout
       ↓
Dense 256 + BatchNorm + ReLU + Dropout
       ↓
Dense 128 + BatchNorm + ReLU
       ↓
29-Class Softmax Output
```

## Dataset

The project processes an ASL alphabet image dataset containing:

- 29 classes
- 63,677 processed samples
- Single-hand static images

## Results

The recorded training run achieved a best validation accuracy of approximately **99.69%**.

The notebooks also evaluate the model using a held-out test split and generate accuracy/loss curves and a confusion matrix.

> Note: The reported result reflects the dataset and evaluation methodology used in this project and should not be interpreted as real-world generalization performance.

## Technologies

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

## Getting Started

Clone the repository:

```bash
git clone https://github.com/ASAD001-P/SignApp.git
```

Open the notebooks in Google Colab or Jupyter Notebook and run them sequentially.

The first notebook performs hand landmark extraction and normalization. The second performs feature engineering, model training, evaluation, and visualization.

## License

This project is licensed under the MIT License.
