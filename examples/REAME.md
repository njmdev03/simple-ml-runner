# Examples

This directory contains example configurations to get you started with some common datasets.

Each directory is outlined below, with more details and analysis in each directories README.

Every example include multiple configs. One base config defines the dataset and common parameters for that dataset, then separate configs define multiple model architectures. Every dataset includes a basic MLP and CNN implementation found in the `./shared` directory. The image datasets (MNIST, Patch Camelyon, and CIFAR-100) additionally include a Vision Transformer (ViT) model. Effectiveness for each dataset and model can be found in their respective READMEs.

## Folders

### 0. Shared

The shared folder includes the base config used by all other examples. This config enables the use of a cuda device for training and inference if the correct version of pytorch is installed and the machine has a GPU supporting CUDA. It additionally defines the Optimizer, Loss function, and learning rate used by all other configs. Note that the optimizer used is a simple Adam optimizer with no decay or other special parameters, so performance on many datasets will be lower than technically possible.

These examples are not intended to make the best models possible for each dataset, they are only intended to show off the difference between different methods and datasets, and how to use the tool.

### 1. MNIST (Handwritten Digits)

Type: Image Classification (Grayscale)

60,000 28x28 grayscale images of handwritten digits with 10 classes.

MNIST is a very common and old dataset for machine learning. These days it is very easy to make a model that identifies these digits with a very high accuracy. Examples are provided using an MLP, CNN, and ViT model.

- **MLP (Simple):** `examples/MNIST/mnist_shared_mlp.py`
- **CNN (Advanced):** `examples/MNIST/mnist_shared_cnn.py`

### 2. CIFAR-100 (Object Recognition)

Type: Image Classification

60,000 32x32 color (3 channel) images with 100 classes.

CIFAR-100 is the most complex dataset example in this repo. It has many more classes and much more input data than other datasets, so it uses a much larger MLP and CNN model than the other examples to account for this fact.

- **MLP:** `examples/CIFAR-100/cifar100_shared_mlp.py`
- **CNN:** `examples/CIFAR-100/cifar100_shared_cnn.py`

### 3. PatchCamelyon (Medical Imaging)

Type: Binary Classification

327,680 96x96 color (3 channel) patches (images) with 2 classes.

Patch Camelyon is a dataset of medical images with the goal being to identify the presence of a tumor in the image. This is a large dataset, but relatively simple to train since it only contains two output classes.

- **MLP:** `examples/PatchCamelyon/pcam_shared_mlp.py`
- **CNN:** `examples/PatchCamelyon/pcam_shared_cnn.py`

### 4. UCI Adult Income (Census Data)

Type: Tabular Prediction

48,842 rows of 14 attributes, mixed between continous and catagorical with some missing data points. 2 classes (<=50K income, >50K income)

The UCI Adult Income data is based on US Census data, with the goal being to predict wether an individual will make more than 50K a year. The dataset is tabular unlike the other examples, and includes missing datapoints which are cleaned up in the custom Dataset object. Notably, this dataset includes a CNN configuration, showing how tabular datasets can be shaped to be run through CNNs normally intended for image data.

- **MLP:** `examples/UCIAdultIncome/adult_shared_mlp.py`
- **CNN:** `examples/UCIAdultIncome/adult_shared_cnn.py`
  - Note: This demonstrates how to reshape tabular data to fit into a CNN architecture.
