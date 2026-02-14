# CIFAR-100 Image Classification

The CIFAR-100 dataset consists of 60,000 32x32 color images in 100 classes. There are 50,000 training images and 10,000 test images.

## Configurations

All configurations inherit from `cifar100_base.py`, which includes data augmentation (RandomHorizontalFlip, RandomCrop) and normalization.

### 1. Multi-Layer Perceptron (MLP)
- **Config:** `cifar100_mlp.py`
- **Details:** Uses a larger MLP (3072 input features) to handle the complexity, but accuracy remains low (<30%) due to the dataset's high dimensionality and lack of spatial awareness in MLPs.
- **Command:**
  ```bash
  cd examples/CIFAR-100
  python ../../main.py job --config cifar100_mlp.py
  ```

### 2. Convolutional Neural Network (CNN)
- **Config:** `cifar100_cnn.py`
- **Details:** Significantly outperforms the MLP by leveraging convolutional layers. Achieves much higher accuracy by identifying local patterns in the color images.
- **Command:**
  ```bash
  cd examples/CIFAR-100
  python ../../main.py job --config cifar100_cnn.py
  ```

### 3. Vision Transformer (ViT)
- **Config:** `cifar100_vit.py`
- **Details:** A more complex ViT model with 128 embedding dimensions and 6 layers, demonstrating how transformers can scale to more complex image tasks.
- **Command:**
  ```bash
  cd examples/CIFAR-100
  python ../../main.py job --config cifar100_vit.py
  ```

## Visualizations

Visualize the training performance and model architecture:
```bash
python ../../main.py vis --config cifar100_cnn.py
```

### Metrics & Predictions
| Training Progress | Sample Predictions |
| :---: | :---: |
| ![CIFAR-100 Loss/Accuracy](../../images/cifar100_combined.png) | ![CIFAR-100 Predictions](../../images/cifar100_samples.png) |

### Model Architecture
![CIFAR-100 Architecture](../../images/cifar100_model.png)
