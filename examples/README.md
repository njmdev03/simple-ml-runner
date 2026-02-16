# Examples

This directory contains example configurations to get you started with some common datasets. Each example demonstrates different model architectures applied to various data types (image, tabular).

These examples are designed to run on GPUs with 8GB of vram or more. While this is not necessary for any provided datasets or models, running on lesser hardware will require some changes to the hyperparameters. If you need to change this, you likely only need to lower the `TESTING_BATCH_SIZE`, which will slow down the testing phase of each example.

## Overview

Each example directory follows a consistent structure:

- `*_base.py`: Defines the dataset, transforms, and common parameters.
- `*_mlp.py`: Configuration for a Multi-Layer Perceptron.
- `*_cnn.py`: Configuration for a Convolutional Neural Network.
- `*_vit.py`: Configuration for a Vision Transformer (image datasets only).

All examples inherit from a shared base configuration in the `shared/` directory.

## Models

Three models were chosen for the examples. Multi-Layer-Perceptron (MLP), Convolutional Neural Network (CNN), and Vision Transformers (ViT).

MLPs perform very well on simple datasets with few classes. Two great examples of this are the MNIST dataset and UCI Adult Income. These networks are very simple to create, perform well, and are easy to compute, however, they do not perform as well on more complicated datasets, particularly those containing images. CIFAR-100 and Patch Camelyon are great examples of this. With CIFAR-100 containing 100 different classes for the network to differentiate, a simple MLP really struggles and has to be scaled up to sizes that are difficult to run in order to get any performance. Patch Camelyon is similar, but mostly because the images are quite large with small details differentiating them.

CNNs perform much better on the image datasets. While it is possible to convert tabular data to something a CNN can process (as shown with UCI Adult Income), it is not recommended, since CNNs are designed around the fact the neighboring pixels are related, which may or may not be true when converting tabular data. The CNNs perform well on all of the provided datasets.

Vision Transformers are provided to show off a more modern setup. They should perform very well on complex image datasets, however they are much more difficult to train, requiring many more epochs than other models and more complex training setups. The provided examples still have a lot of room for improvement, but the basic ideas are present in the examples.

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
