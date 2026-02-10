# ML Job Runner

A flexible framework for creating, training, and testing machine learning models from config files or the command line.

## Configuration Reference

The following configuration keys can be used in your JSON, YAML, TOML, or Python config files. They can also be overridden by environment variables or command-line arguments.

| Key | Type | Default | CLI Argument | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Meta** | | | | |
| `SILENT` | `bool` | `False` | `--silent` | Suppress console output. |
| `PROFILE` | `bool` | `False` | `--profile` | Enable performance profiling. |
| `PROFILE_OUTPUT` | `str` | `None` | `--profile-output` | Path to save profiling report. |
| `PROFILE_DIR` | `str` | `''` | | Directory to search for profiling logs (vis only). |
| `PROFILE_NAME` | `str` | `''` | | Name of profiling log file (vis only). |
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
| `SAVE_METADATA` | `bool` | `True` | `--save-metadata` | Save JSON metadata with checkpoints. |
| `RESUME` | `bool/str` | `False` | `--resume` | Resume training from checkpoint (path or bool). |
| **Early Halting** | | | | |
| `EARLY_HALT_CONDITION`| `str` | `'None'` | `--early-halt-condition`| Condition to stop early ('Loss' or 'Accuracy'). |
| `EARLY_HALT_THRESHOLD`| `float` | `0.0` | `--early-halt-threshold`| Threshold for early halting. |
| **Testing** | | | | |
| `TESTING_BATCH_SIZE` | `int` | `32` | `--testing-batch-size` | Batch size for testing (defaults to BATCH_SIZE). |
| `TESTING_CRITERION` | `list` | `[]` | `--testing-criterion` | List of criteria to evaluate (e.g., `['CrossEntropyLoss']`). |
| `TEST_ON_TRAINING_DATA`| `bool` | `False` | `--test-on-training-data` | Run evaluation on training data. |
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

## Environment Variables
Any configuration key can be set via an environment variable with the same name.
Example: `export EPOCHS=50`

## CLI Arguments
CLI arguments override individual config keys. See `python main.py --help` for a full list.
