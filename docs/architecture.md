# Project Architecture
`ml-runner` is designed to be highly modular and configuration-driven, separating model definitions, data handling, and the training loop.

## Key Components

### 1. Runner Engine (`src/ml_runner/core/runner.py`)
The engine is the heart of the project. It orchestrates the lifecycle of a machine learning experiment.
- **Initialization**: Loads configuration, sets up devices (CPU/GPU), and initializes models and datasets.
- **Training Loop**: Iterates through epochs, handles backpropagation, and invokes optimizers.
- **Evaluation**: Periodically runs evaluation on the validation/test sets.
- **Checkpointing**: Automatically saves model states at regular intervals or when performance improves.

### 2. Configuration System (`src/ml_runner/core/config/`)
Experiments are defined in YAML files. The system uses **Pydantic** to validate these configurations against a strictly defined schema (`schema.py`).
- **Overrides**: Parameters can be overridden via command-line arguments.
- **Merging**: Multiple YAML files can be merged (e.g., a base config + an experiment-specific config).

### 3. Registries (`src/ml_runner/core/registries.py`)
A central registry system allows the runner to dynamically discover and instantiate:
- **Models**: Registered with `@model_registry.register`.
- **Datasets**: Registered with `@dataset_registry.register`.
- **Tasks**: Defines how inputs and outputs are processed for specific ML problems (e.g., Classification, Translation).

### 4. Extensions & Exporters (`src/ml_runner/extensions/`, `src/ml_runner/core/exporters/`)
- **Extensions**: Add functionality like logging (TensorBoard), profiling, and custom event handlers.
- **Exporters**: Transform experiment results into other formats (e.g., ONNX, plots, metrics).

## Workflow

1. **CLI Call**: User runs `ml-runner run -c my_config.yml`.
2. **Parsing**: CLI arguments and YAML files are merged and validated.
3. **Registry Lookup**: The runner finds the requested model and dataset.
4. **Execution**: The training engine starts the job.
5. **Logging**: Results are written to the `output/` directory as the job progresses.
