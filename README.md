# Simple ML Runner

> **A beginner-friendly tool for training and testing PyTorch models without the boilerplate.**

Welcome to `simple-ml-runner`! This project is designed to help students and researchers focus on **designing models** and **analyzing results**, rather than writing the same training loops over and over again.

## 🚀 How It Works

Traditional machine learning projects require writing code to load data, iterate through epochs, calculate loss, backpropagate errors, save checkpoints, and log statistics.

**Simple ML Runner handles all of that for you.**

You simply provide a **Configuration File** (a Python script) that defines:
1.  **The Model**: Your neural network architecture.
2.  **The Data**: Where your training and testing data comes from.
3.  **The Hyperparameters**: Settings like learning rate, batch size, and epochs.

The runner takes this config and executes the entire **Job** (Training + Testing), saving all results and models automatically.

---

## 📦 Installation & Setup

All you need is Python and the project dependencies.

1.  **Create a Virtual Environment** (Recommended):
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚡ Quick Start: MNIST Example

The "Hello World" of Machine Learning is the **MNIST** dataset (handwritten digits). Let's train a simple neural network to recognize them.

**Run the job:**
```bash
python main.py job --config examples/MNIST/mnist_shared_mlp.py
```

**What happens?**
1.  The dataset is downloaded automatically.
2.  The model (a Multi-Layer Perceptron) is initialized.
3.  Training starts for the defined number of epochs.
4.  Results are saved to `results/` and checkpoints to `checkpoints/`.

---

## 📚 Running Examples

We have included several example configurations to help you get started with different types of data.

### 1. MNIST (Handwritten Digits)
*Type: Image Classification (Grayscale)*

-   **MLP (Simple):** `examples/MNIST/mnist_shared_mlp.py`
-   **CNN (Advanced):** `examples/MNIST/mnist_shared_cnn.py`

### 2. CIFAR-100 (Object Recognition)
*Type: Image Classification (Color, 100 classes)*

-   **MLP:** `examples/CIFAR-100/cifar100_shared_mlp.py`
-   **CNN:** `examples/CIFAR-100/cifar100_shared_cnn.py`

### 3. PatchCamelyon (Medical Imaging)
*Type: Binary Classification (Tumor detection)*

-   **MLP:** `examples/PatchCamelyon/pcam_shared_mlp.py`
-   **CNN:** `examples/PatchCamelyon/pcam_shared_cnn.py`

### 4. UCI Adult Income (Census Data)
*Type: Tabular Prediction (>50k income)*

-   **MLP:** `examples/UCIAdultIncome/adult_shared_mlp.py`
-   **CNN (Experimental):** `examples/UCIAdultIncome/adult_shared_cnn.py`
    *   *Note: This demonstrates how to reshape tabular data to fit into a CNN architecture.*

---

## 📊 Visualization

After training, you can visualize the results (Loss curves, Accuracy, Model Architecture, etc.) without writing any plotting code.

**Generate Visualizations:**
```bash
python main.py vis --config examples/MNIST/mnist_shared_mlp.py
```

### Example Outputs

| Loss Curve | Accuracy Curve |
| :---: | :---: |
| ![Loss Curve Placeholder](docs/placeholders/loss_curve.png) | ![Accuracy Curve Placeholder](docs/placeholders/accuracy_curve.png) |

| Sample Predictions | Model Architecture |
| :---: | :---: |
| ![Predictions Placeholder](docs/placeholders/predictions.png) | ![Architecture Placeholder](docs/placeholders/architecture.png) |

---

## 🛠️ Configuration Reference

You can customize your experiments by modifying the config files. Here are the most common settings:

| Setting | Description | Example |
| :--- | :--- | :--- |
| `MODEL` | The neural network object to train. | `MLP(784, 10)` |
| `EPOCHS` | How many times to iterate over the dataset. | `10` |
| `BATCH_SIZE` | Number of samples processed at once. | `64` |
| `LEARNING_RATE` | How fast the model updates its weights. | `0.001` |
| `TRAIN_DATASET` | The PyTorch dataset for training. | `datasets.MNIST(...)` |
| `EARLY_HALT_CONDITION` | Stop training if metric stops improving. | `"Accuracy"` |

### Advanced Usage: Inheritance
You can inherit settings from other config files to avoid repetition!
```python
# my_experiment.py
CONFIG = "base_config.py"  # Load defaults from here

LEARNING_RATE = 0.01       # Override specific value
```

## Modes

The first cli argument when executing the tool will decide what mode it runs in. In general, `job` and `vis` will be the most common, but you can read about all available modes below.

### Training



### Testing



### Job



### Batch



### Visualization



## Configuration Reference

The following configuration keys can be used in your JSON, YAML, TOML, or Python config files. They can also be overridden by environment variables or command-line arguments.

| Key | Type | Default | CLI Argument | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Meta** | | | | |
| `SILENT` | `bool` | `False` | `--silent` | Suppress console output. |
| `PROFILE` | `bool` | `False` | `--profile` | Enable performance profiling. |
| `PROFILE_OUTPUT` | `str` | `None` | `--profile-output` | Path to save profiling report. |
<!-- | `PROFILE_DIR` | `str` | `''` | | Directory to search for profiling logs (vis only). |
| `PROFILE_NAME` | `str` | `''` | | Name of profiling log file (vis only). | -->
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

CLI arguments override individual config keys. See `python main.py --help` for a full list.