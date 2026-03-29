# Simple ML Runner

> **A beginner-friendly tool for training and testing PyTorch models without the boilerplate.**

Welcome to `simple-ml-runner`! This project helps students and researchers focus on model design and analysis rather than repetitive training loops.

## 🚀 How It Works

Simple ML Runner automates the machine learning lifecycle. You define your experiment in a YAML configuration file, and the runner handles:

- **Data Loading**: Dataset downloading and preprocessing.
- **Training**: Epoch loops, backpropagation, and optimization.
- **Evaluation**: Calculating metrics like BLEU or Perplexity.
- **Artifacts**: Saving checkpoints, logs, and plots automatically.

## 📦 Installation

1. **Create a Virtual Environment**:

    ```bash
    python -m venv .venv
    # Windows: .venv\Scripts\activate | Unix: source .venv/bin/activate
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: For GPU support, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/).*

## 🛠️ Basic Usage

Run a training job using a configuration file:

```bash
ml-runner run -c examples/MNIST/mnist_mlp.yml
```

Export plots for an experiment:

```bash
ml-runner export plot -c examples/nlp/gru_mt_onehot.yml
```

Override parameters from the command line:

```bash
ml-runner run -c examples/MNIST/mnist_mlp.yml --lr 0.01 --epochs 10
```

## 📚 Documentation

For more detailed guides and NLP tutorials, see the [docs/](docs/) folder:

- [NLP Examples Guide](docs/examples/nlp_guide.md)
- [Project Architecture](docs/architecture.md)
