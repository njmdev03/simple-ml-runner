import argparse
import os

from config.manager import ConfigManager
from engine.loader import load_from_pyscript
from engine.compute_transforms import compute_stats
from engine.job_runner import run_job

def parse_args():
    parser = argparse.ArgumentParser(description="ML Job Runner - Train and Test ML models via config.")

    parser.add_argument("operation",
                        choices=['batch', 'job', 'test', 'train', 'vis', 'stats'],
                        help="Which supported operation to run. Supported values: batch, job, test, train, vis, stats")

    # Meta Options
    parser.add_argument("--config", "-c", action="append", help="Config file(s) to load. Loaded as overrides in batch mode.")
    parser.add_argument("--jobs", "-j", action="append", help="Config file(s) to load as jobs in batch mode.")
    parser.add_argument("--silent", action="store_true", help="Print less to console")

    # Behavior Flags
    parser.add_argument("--dont-train", action="store_true", help="Disable training process.")
    parser.add_argument("--train", action="store_true", help="Enable training process (Default).")
    parser.add_argument("--dont-test", action="store_true", help="Disable testing process.")
    parser.add_argument("--test", action="store_true", help="Enable testing process (Default).")

    # Model and Datasets
    parser.add_argument("--model", help="Path to a python file containing the neural network blueprint.")
    parser.add_argument("--train-dataset", help="Path to Python file containing a reference to the training dataset.")
    parser.add_argument("--test-dataset", help="Path to a Python file containing a reference to the testing dataset.")
    parser.add_argument("--devices", help="Comma-separated list of devices to run models on, ordered by priority.")

    # Training Options
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--epochs", type=int, help="Number of epochs.")
    parser.add_argument("--optimizer", help="Optimizer name or reference.")
    parser.add_argument("--train-criterion", help="Training criterion name or reference.")
    parser.add_argument("--early-halt-condition", choices=['None', 'Loss', 'Accuracy'], help="Condition to exit early.")
    parser.add_argument("--early-halt-threshold", type=float, help="Threshold for early halting.")
    parser.add_argument("--test-while-training", action="store_true", help="Evaluate model after each epoch. Follows testing options")
    parser.add_argument("--final-output-path", help="Path to save the model after training.")

    # Testing Options
    parser.add_argument("--testing-batch-size", type=int, help="Testing batch size.")
    parser.add_argument("--test-on-training-data", action="store_true", help="Run tests on training data.")
    parser.add_argument("--testing-criterion", action="append", help="List of testing criteria.")
    parser.add_argument("--test-checkpoints", action="store_true", help="Run tests on all checkpoints.")
    parser.add_argument("--save-tests", help="Path to save test results (extension determines format, CSV or XLSX supported).")

    # Checkpoint Options
    parser.add_argument("--checkpoint-frequency", type=int, dest="check_rate", help="Frequency in epochs to save checkpoints.")
    parser.add_argument("--save-metadata", action="store_true", help="Save JSON metadata files alongside checkpoints.")
    parser.add_argument("--checkpoint-dir", dest="check_model_dir", help="Directory to save checkpoints and metadata to.")
    parser.add_argument("--checkpoint-name", dest="check_model_name", help="Template string for checkpoint names.")
    parser.add_argument("--resume", help="Resume from checkpoint (bool or path).")

    # Profiler Options
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling.")
    parser.add_argument("--profile-output", help="Path to save profiling results.")

    # Visualization options

    return parser.parse_args()

def load_config(config_paths, args=None):
    manager = ConfigManager()

    config = manager.load_config_tree(config_paths)
    config = manager.apply_env_overrides(config)
    config = manager.apply_cli_overrides(config, args)

    return config

def main():
    # Parse the passed arguments
    args = parse_args()

    # Parse the passed configs
    config_paths = args.config if args.config else []
    config = load_config(config_paths, args=args)

    match args.operation:
        case 'batch':
            # Iterate over each job config file and run it
            for conf in config.jobs:
                job = load_config([conf, config_paths], args=args)
                run_job(job)

        case 'job':
            # Parse the configs as a single training and testing operation
            run_job(config)

        case 'test':
            # Only test the model, ignore 'TRAIN' and 'TEST' option
            config["TEST"] = True
            config["TRAIN"] = False

            run_job(config)

        case 'train':
            # Only train the model, ignore 'TRAIN' option, 'TEST' can still be used to bypass 'TEST_WHILE_TRAINING'
            config["TRAIN"] = True

            run_job(config)

        case 'vis' 'visualize':
            # Visualize the model, training process, accuracy, etc.
            # TODO
            pass
        case 'stats':
            # Compute transform stats for the dataset found in the config
            # Make sure that no transforms are provided when running stats
            # After running stats, the values should be included in the config's
            # dataset transforms.
            train_dataset = None
            ds_val = config.get('TRAIN_DATASET')
            if ds_val:
                if isinstance(ds_val, (str, os.PathLike)):
                    train_dataset = load_from_pyscript(ds_val, 'TRAIN_DATASET')
                else:
                    train_dataset = ds_val

            mean, std = compute_stats(train_dataset, batch_size=config.get('BATCH_SIZE'))

            print("mean:", mean)
            print("std:", std)

        case _:
            # Unsupported operation passed, exit.
            # TODO: Add a suggested command based on heuristics of the passed string.
            print(f"Operation '{args.operation}' is not supported.")
            quit(1)

if __name__ == "__main__":
    main()
