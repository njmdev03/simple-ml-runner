# Examples

This directory contains example configurations to get you started with some common datasets. Each example demonstrates different model architectures applied to various data types (image, tabular).

## Overview

Each example directory follows a consistent structure:
- `*_base.py`: Defines the dataset, transforms, and common parameters.
- `*_mlp.py`: Configuration for a Multi-Layer Perceptron.
- `*_cnn.py`: Configuration for a Convolutional Neural Network.
- `*_vit.py`: Configuration for a Vision Transformer (image datasets only).

All examples inherit from a shared base configuration in the `shared/` directory.

## Example Datasets

### 1. [MNIST (Handwritten Digits)](./MNIST/README.md)
**Type:** Image Classification (Grayscale)
60,000 28x28 grayscale images of handwritten digits (0-9).

### 2. [CIFAR-100 (Object Recognition)](./CIFAR-100/README.md)
**Type:** Image Classification (RGB)
60,000 32x32 color images with 100 classes.

### 3. [PatchCamelyon (Medical Imaging)](./PatchCamelyon/README.md)
**Type:** Binary Classification
327,680 96x96 color patches for tumor detection.

### 4. [UCI Adult Income (Census Data)](./UCIAdultIncome/README.md)
**Type:** Tabular Prediction
48,842 rows of census data to predict income brackets (>50K or <=50K).

---

## Shared Configuration

The `shared/` folder contains models and base configurations used across all examples:
- **Optimizer:** Adam (default, no decay).
- **Loss Function:** CrossEntropyLoss.
- **Hardware:** Auto-detects CUDA for GPU acceleration if available.

Note: These examples are intended for educational purposes and show-casing the tool, not for achieving state-of-the-art performance.
