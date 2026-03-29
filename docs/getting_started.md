# Getting Started with ML-Job-Runner

Welcome to the documentation for `ml-runner`! This project is designed for students and researchers looking for an educational, yet powerful, framework for PyTorch experiments.

## Quick Start

1. **Installation**:

   ```bash
   pip install -r requirements.txt
   ```

2. **First Experiment**:
   Run an MNIST classification job:

   ```bash
   ml-runner run -c examples/MNIST/mnist_mlp.yml
   ```

3. **View Results**:
   All logs and plots are saved in `examples/output/`.

## Guided Tutorials

- **[NLP Examples Guide](examples/nlp_guide.md)**: Explore Recurrent Neural Networks (RNN, LSTM, GRU) for translation and text generation.
- **[Custom Models & Datasets](customization.md)**: Learn how to register your own PyTorch code with the runner.
- **Project Architecture**: Learn about the modular design in [architecture.md](architecture.md).

## Configuration Basics

All experiments are driven by YAML files. A typical config includes:

- `task`: The type of problem (e.g., `TranslationTask`).
- `model`: Architecture parameters like `hidden_dim` and `n_layers`.
- `dataset`: Path to the data and transformation settings.
- `training`: Epochs, optimizer, and learning rate.

Example (snippet):

```yaml
training:
  enabled: true
  epochs: 5
  optimizer:
    lr: 0.001
```

## Analysis & Visualization

Use the `export` command to generate plots from your experiment data:

```bash
ml-runner export plot -c examples/nlp/gru_mt_onehot.yml
```

This will create `loss.png`, `bleu.png`, and other metrics in the `plots/` subdirectory of your output folder.
