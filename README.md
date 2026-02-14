# Simple ML Runner

> **A beginner-friendly tool for training and testing PyTorch models without the boilerplate.**

Welcome to `simple-ml-runner`! This project is designed to help students and researchers focus on designing models and analyzing results, rather than writing the same training loops over and over again.

## WARNING!!!

This project dynamically loads python files in order to create models and datasets. This is incredibly powerful for an educational tool like this, since it allows for just about any model architecture you can write in Pytorch, as well as custom datasets, optimizers, and loss functions.

**However**, dynamically executing arbitrary python code can be dangerous. Only run configs you understand 100% and never run configs from the internet.

## 🚀 How It Works

Traditional machine learning projects require writing code to load data, iterate through epochs, calculate loss, backpropagate errors, save checkpoints, and log statistics.

**Simple ML Runner handles all of that for you.**

You simply provide a Configuration File that defines:

1. **The Model**: Your neural network architecture.
2. **The Data**: Where your training and testing data comes from.
3. **The Hyperparameters**: Settings like learning rate, batch size, and epochs.

The runner takes this config and executes the entire Job (Training + Testing), saving all results and models automatically.

---

## 📦 Installation & Setup

All you need is Python and the project dependencies.

1. **Create a Virtual Environment** (Recommended):

    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. **Install Dependencies**:

    All dependencies can be installed from requirements.txt, but if you have a CUDA enabled GPU (basically anything from Nvidia) pytorch and torchvision should be installed separately according to pytorch's getting started guide.

    Go to [Getting Started](https://pytorch.org/get-started/locally/) and select your operation system, pip, python, and the desired CUDA version (any should work on the provided examples), then run the provided command.

    After installing Pytorch with CUDA support, or if you do not have a CUDA GPU, run

    ```bash
    pip install -r requirements.txt
    ```

    To install the remaining requirements. Some examples have additional requirements used for fetching datasets, if you want to run those additionally run 

    ```bash
    pip install -r ./examples/requirements.txt
    ```

---

## ⚡ Quick Start: MNIST Example

The "Hello World" of Machine Learning is the MNIST dataset (handwritten digits). Let's run the provided example to train a model!

### Run the job

It is recommended to invoke the main job from the example's directory since config paths are relative to the directory you execute `main.py` from, not where the config is stored. So first open the directory containing the MNIST examples, then run the job.

```bash
cd examples/MNIST
python ../../main.py job --config mnist_shared_mlp.py
```

### What happens?

1. The dataset is downloaded automatically.
2. The model (a Multi-Layer Perceptron) is initialized.
3. Training starts for the defined number of epochs.
4. Results are saved to `results/` and checkpoints to `checkpoints/`.

---

## 📊 Visualization

After training, you can visualize the results (Loss curves, Accuracy, Model Architecture, etc.) without writing any plotting code.

Simply run

```bash
python ../../main.py vis --config mnist_shared_mlp.py
```

And a collection of chart images will be output to the `vis/mlp/` directory

### Example Outputs

| Loss Curve | Accuracy Curve |
| :---: | :---: |
| ![Loss Curve Placeholder](docs/placeholders/loss_curve.png) | ![Accuracy Curve Placeholder](docs/placeholders/accuracy_curve.png) |

| Sample Predictions | Model Architecture |
| :---: | :---: |
| ![Predictions Placeholder](docs/placeholders/predictions.png) | ![Architecture Placeholder](docs/placeholders/architecture.png) |

---

## 📚 Running Examples

Several example configurations are included to help you get started with different types of data and models. For detailed analysis, performance results, and run instructions for each dataset, see the [Examples Directory](./examples/README.md).

### 1. [MNIST (Handwritten Digits)](./examples/MNIST/README.md)
Type: Image Classification (Grayscale)
- **MLP (Simple):** `examples/MNIST/mnist_mlp.py`
- **CNN (Advanced):** `examples/MNIST/mnist_cnn.py`
- **ViT (Modern):** `examples/MNIST/mnist_vit.py`

### 2. [CIFAR-100 (Object Recognition)](./examples/CIFAR-100/README.md)
Type: Image Classification (Color, 100 classes)
- **MLP:** `examples/CIFAR-100/cifar100_mlp.py`
- **CNN:** `examples/CIFAR-100/cifar100_cnn.py`
- **ViT:** `examples/CIFAR-100/cifar100_vit.py`

### 3. [PatchCamelyon (Medical Imaging)](./examples/PatchCamelyon/README.md)
Type: Binary Classification (Tumor detection)
- **MLP:** `examples/PatchCamelyon/pcam_mlp.py`
- **CNN:** `examples/PatchCamelyon/pcam_cnn.py`
- **ViT:** `examples/PatchCamelyon/pcam_vit.py`

### 4. [UCI Adult Income (Census Data)](./examples/UCIAdultIncome/README.md)
Type: Tabular Prediction (>50k income)
- **MLP:** `examples/UCIAdultIncome/adult_mlp.py`
- **CNN:** `examples/UCIAdultIncome/adult_cnn.py`
  - Note: Demonstrates reshaping tabular data for CNNs.

---

## 🛠️ Configuration

You can customize the examples or write your own experiments using using config files. The models and datasets are intentionally defined as pytorch Model and Dataset objects to give you full flexibility in your experiments. Your config files can be a Python, JSON, YAML, TOML, or INI file, just note that when not using a Python file you will need to point your configs to one or more Python files containing your Model and Dataset objects.

For complete configuration documentation, see [Configuration Reference](#configuration-reference)

### Advanced Usage: Inheritance

You can inherit settings from other config files to avoid repetition!

```python
# my_experiment.py
CONFIG = "base_config.py"  # Load defaults from here

LEARNING_RATE = 0.01       # Override value
```

All settings can also be overridden by environment variables or command line arguments for quick testing.

## Modes

The first cli argument when executing the tool will decide what mode it runs in. In general, `job` and `vis` will be the most common, but you can read about all available modes below.

### Training

```bash
python main.py train [..args]
```

In training mode the tool will ignore the `TRAIN` configuration and `--dont-train` flags and always run the training phase.

### Testing

```bash
python main.py test [..args]
```

In testing mode the tool will ignore the `TEST` configuration and `--dont-test` flags and always run the testing phase.

### Job

```bash
python main.py job [..args]
```

Job mode is the most commonly used. It sequentially runs the training and testing phases together. Either phase can be bypassed using the appropriate CLI argument (`--dont-train`, `--dont-test`), which can be useful if an experiment gets interrupted.

### Batch

```bash
python main.py batch [..args]
```

In batch mode the tool will sequentially run a series of configs in job mode. Add jobs to be run by passing their configs with the `--jobs` argument. You can use the `--jobs` argument multiple times to add every config you want to run.

### Visualization

```bash
python main.py vis [..args]
```

Running in visualization mode will use the output from enabling `SAVE_TESTS` and/or `PROFILE` to generate graphs of training results, model architecture, and examples of model inference. 

Use the `SHOW` key to open windows to view these graphs, or provide a `VIS_OUTPUT_DIR` to save images automatically. Use `VIS_METRICS` to choose what graphs to output, including `model` to view your model architecture, `samples` to view examples of your models predictions on your testing dataset, and `combined` to create a graph showing the loss and accuracy of your model on each of your datasets, all in one convenient graph!

### Stats

```bash
python main.py stats [..args]
```

Run your config in stats mode to find the mean and standard deviation of your training dataset. Make sure to not apply a transform based on these values when running `stats`, but once you have the mean and standard deviation you can add them to your config as a transform to improve training (check out the examples to see this in action!)

## Configuration Reference

The following configuration keys can be used in your JSON, YAML, TOML, or Python config files. They can also be overridden by environment variables or command-line arguments.

| Key | Type | Default | CLI Argument | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Meta** | | | | |
| `SILENT` | `bool` | `False` | `--silent` | Suppress console output. |
| `PROFILE` | `bool` | `False` | `--profile` | Enable performance profiling. |
| `PROFILE_OUTPUT` | `str` | `None` | `--profile-output` | Path to save profiling report. |
| **Model & Device** | | | | |
| `MODEL` | `str` | `None` | `--model` | Path to python file defining the model (must contain `MODEL` or `Net` object). |
| `DEVICES` | `list` | `['cpu']` | `--devices` | List of devices to use (e.g., `['cuda', 'cpu']`). |
| **Data** | | | | |
| `TRAIN_DATASET` | `str` | `None` | `--train-dataset` | Path to python file defining training dataset. |
| `TEST_DATASET` | `str` | `None` | `--test-dataset` | Path to python file defining testing dataset. |
| `FINAL_OUTPUT_PATH` | `str` | `'model_final.pt'` | `--final-output-path` | Path to save the trained model. |
| **Training Flags** | | | | |
| `TRAIN` | `bool` | `True` | `--train`/`--dont-train` | Enable/Disable training phase. |
| `TEST` | `bool` | `True` | `--test`/`--dont-test` | Enable/Disable testing phase. |
| **Hyperparameters** | | | | |
| `BATCH_SIZE` | `int` | `32` | `--batch-size` | Batch size for training. |
| `LEARNING_RATE` | `float` | `0.001` | `--lr` | Learning rate for optimizer. |
| `EPOCHS` | `int` | `10` | `--epochs` | Number of training epochs. |
| `OPTIMIZER` | `str` | `'Adam'` | `--optimizer` | Optimizer name (e.g., 'Adam', 'SGD'). |
| `TRAIN_CRITERION` | `str` | `'CrossEntropyLoss'` | `--train-criterion` | Loss function for training. |
| **Checkpointing** | | | | |
| `CHECK_RATE` | `int` | `1` | `--checkpoint-frequency` | Save checkpoint every N epochs. |
| `CHECK_MODEL_DIR` | `str` | `'checkpoints/'` | `--checkpoint-dir` | Directory to save checkpoints. |
| `CHECK_MODEL_NAME` | `str` | `'model_epoch_$epoch'` | `--checkpoint-name` | Template for checkpoint filenames. |
| `SAVE_METADATA` | `bool` | `True` | `--dont-save-metadata` | Save JSON metadata with checkpoints. |
| `RESUME` | `bool/str` | `False` | `--resume` | Resume training from checkpoint (path or bool). |
| **Early Halting** | | | | |
| `EARLY_HALT_CONDITION` | `str` | `'None'` | `--early-halt-condition`| Condition to stop early ('Loss' or 'Accuracy'). |
| `EARLY_HALT_THRESHOLD` | `float` | `0.0` | `--early-halt-threshold`| Threshold for early halting. |
| **Testing** | | | | |
| `TESTING_BATCH_SIZE` | `int` | `32` | `--testing-batch-size` | Batch size for testing (defaults to BATCH_SIZE). |
| `TESTING_CRITERION` | `list` | `[]` | `--testing-criterion` | List of criteria to evaluate (e.g., `['CrossEntropyLoss']`). |
| `TEST_ON_TRAINING_DATA` | `bool` | `False` | `--test-on-training-data` | Run evaluation on training data. |
| `TEST_WHILE_TRAINING` | `bool` | `False` | `--test-while-training` | Run evaluation after every epoch. |
| `TEST_CHECKPOINTS` | `bool` | `False` | `--test-checkpoints` | Evaluate all saved checkpoints. |
| `SAVE_TESTS` | `str` | `None` | `--save-tests` | Path to save test results (CSV/XLSX). |
| **Visualization** | | | | |
| `VIS_TYPE` | `list` | `['all']` | `--vis-type` | Visualization types ('all', 'loss', 'accuracy', etc.). |
| `VIS_OUTPUT_DIR` | `str` | `'vis'` | `--vis-output-dir` | Directory to save plots. |
| `VIS_FORMAT` | `str` | `'png'` | `--vis-format` | Output format (png, jpg, pdf). |
| `VIS_LAYOUT` | `str` | `'individual'` | `--vis-layout` | Layout ('individual', 'grid'). |
| `SHOW` | `bool` | `False` | `--show` | Show plots interactively. |
| `NUM_SAMPLES` | `int` | `10` | `--num-samples` | Number of samples for prediction visualization. |
| `VIS_DATASETS` | `list` | `['all']` | `--vis-datasets` | Filter datasets ('training', 'testing', 'all'). |
| `VIS_METRICS` | `list` | `['all']` | `--vis-metrics` | Filter metrics ('loss', 'accuracy', 'all'). |

### Environment Variables

Any configuration key can be set via an environment variable with the same name.
Example: `export EPOCHS=10`

### CLI Arguments

CLI arguments override their config keys. See `python main.py --help` for a full list of available CLI arguments and their descriptions.

## AI Usage Disclosure

This project and documentation was created with the assistance of AI tools, including Gemini, ChatGPT, Google Antigravity, and GitHub Copilot. 

This project was inspired by a jupyter notebook created with the help of Gemini, which was then translated into a regular python project by hand. That project was later used as the template given to Google Antigravity to create this project, before being enhanced by hand. Most examples were generated by AI tools before being enhanced by hand.
