import argparse
import os
import matplotlib.pyplot as plt

from config.manager import ConfigManager
from engine.loader import load_from_pyscript
from engine.compute_transforms import compute_stats
from engine.job_runner import run_job
from reporting.visualizer import Visualizer

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
    parser.add_argument("--dont-save-metadata", action="store_true", help="Save JSON metadata files alongside checkpoints.")
    parser.add_argument("--checkpoint-dir", dest="check_model_dir", help="Directory to save checkpoints and metadata to.")
    parser.add_argument("--checkpoint-name", dest="check_model_name", help="Template string for checkpoint names.")
    parser.add_argument("--resume", help="Resume from checkpoint (bool or path).")

    # Profiler Options
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling.")
    parser.add_argument("--profile-output", help="Path to save profiling results.")

    # Visualization options
    parser.add_argument("--vis-type", action="append", dest="vis_type",
                        choices=['all', 'loss', 'accuracy', 'combined', 'timing', 'duration', 'samples', 'model'],
                        help="Which visualization(s) to generate. Can be specified multiple times.")
    parser.add_argument("--vis-output-dir", help="Directory to save visualization images (default: vis/).")
    parser.add_argument("--vis-format", help="Output format for visualizations (default: png).")
    parser.add_argument("--vis-datasets", action="append", dest="vis_datasets",
                        choices=['training', 'testing', 'all'],
                        help="Filter by dataset: training, testing, or all. Can be specified multiple times.")
    parser.add_argument("--vis-metrics", action="append", dest="vis_metrics",
                        choices=['loss', 'accuracy', 'all'],
                        help="Filter by metric: loss, accuracy, or all. Can be specified multiple times.")
    parser.add_argument("--vis-layout", choices=['individual', 'grid'],
                        help="Layout for showing plots: 'individual' windows or a single 'grid'. (default: individual)")
    parser.add_argument("--show", action="store_true", help="Display plots interactively instead of saving.")
    parser.add_argument("--num-samples", type=int, help="Number of samples for prediction preview (default: 10).")

    return parser.parse_args()

def load_config(config_paths, args=None):
    manager = ConfigManager()

    config = manager.load_config_tree(config_paths)
    config = manager.apply_env_overrides(config)
    config = manager.apply_cli_overrides(config, args)

    return config

def model_vis(config, visualizer):
    # Visualize model architecture
    model_val = config.get('MODEL')
    show_plots = config.get('SHOW')

    if model_val:
        if isinstance(model_val, (str, os.PathLike)):
            model_obj = load_from_pyscript(model_val, ['MODEL', 'Net'])
        else:
            model_obj = model_val

        if isinstance(model_obj, type):
            model = model_obj()
        else:
            model = model_obj

        visualizer.plot_model_architecture(model, show=show_plots)

def vis_samples(config, visualizer):
    # Load model and dataset for sample predictions
    from engine.job_runner import get_device
    import torch

    device = get_device(config.get('DEVICES'))
    model_val = config.get('MODEL')
    show_plots = config.get('SHOW')
    num_samples = config.get('NUM_SAMPLES')

    if model_val:
        if isinstance(model_val, (str, os.PathLike)):
            model_obj = load_from_pyscript(model_val, ['MODEL', 'Net'])
        else:
            model_obj = model_val

        if isinstance(model_obj, type):
            model = model_obj().to(device)
        else:
            model = model_obj.to(device)

        # Load final model weights if available
        final_path = config.get('FINAL_OUTPUT_PATH')
        if final_path and os.path.exists(final_path):
            model.load_state_dict(torch.load(final_path, map_location=device))

        # Load test dataset
        test_dataset = config.get('TEST_DATASET')
        if test_dataset:
            if isinstance(test_dataset, (str, os.PathLike)):
                test_dataset = load_from_pyscript(test_dataset, 'TEST_DATASET')

            visualizer.plot_sample_predictions(
                model, test_dataset, device,
                num_samples=num_samples, show=show_plots
            )

def main():
    # Parse the passed arguments
    args = parse_args()

    # Parse the passed configs
    config_paths = args.config if args.config else []
    config = load_config(config_paths, args=args)

    # print()
    # print(f"CONFIG DUMP:")
    # print(f"{config}")
    # print()

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

        case 'vis' | 'visualize':
            # Visualize the model, training process, accuracy, etc.
            vis_types = config.get('VIS_TYPE')
            if not isinstance(vis_types, list):
                vis_types = [vis_types]

            output_dir = config.get('VIS_OUTPUT_DIR')
            vis_format = config.get('VIS_FORMAT')
            vis_layout = config.get('VIS_LAYOUT')
            show_plots = config.get('SHOW')

            visualizer = Visualizer(output_dir=output_dir, format=vis_format)

            # Parse filtering options
            vis_datasets_raw = config.get('VIS_DATASETS')
            if not isinstance(vis_datasets_raw, list):
                vis_datasets_raw = [vis_datasets_raw]

            vis_metrics_raw = config.get('VIS_METRICS')
            if not isinstance(vis_metrics_raw, list):
                vis_metrics_raw = [vis_metrics_raw]

            # Convert to capitalized list format for visualizer
            if 'all' in vis_datasets_raw:
                datasets_filter = ['Training', 'Testing']
            else:
                datasets_filter = [d.capitalize() for d in vis_datasets_raw]

            if 'all' in vis_metrics_raw:
                metrics_filter = ['loss', 'accuracy']
            else:
                metrics_filter = vis_metrics_raw

            results_df = None
            profile_df = None

            # Load results CSV from config
            save_tests_path = config.get('SAVE_TESTS')
            if save_tests_path and os.path.exists(save_tests_path):
                results_df = Visualizer.load_results_csv(save_tests_path)
                if not config.get('SILENT'):
                    print(f"Loaded results from {save_tests_path}")

            # Load profile CSV from config
            profile_dir = config.get('PROFILE_DIR')
            profile_name = config.get('PROFILE_NAME')
            profile_path = os.path.join(profile_dir, profile_name) if profile_dir and profile_name else config.get('PROFILE_OUTPUT')
            if profile_path and os.path.exists(profile_path):
                profile_df = Visualizer.load_profile_csv(profile_path)
                if not config.get('SILENT'):
                    print(f"Loaded profile from {profile_path}")

            # Generate requested visualizations
            for vis_type in vis_types:
                match vis_type:
                    case 'all':
                        visualizer.generate_all(results_df, profile_df, show=show_plots,
                                               datasets=datasets_filter, metrics=metrics_filter,
                                               layout=vis_layout)
                        vis_samples(config, visualizer)
                        model_vis(config, visualizer)
                    case 'combined':
                        if results_df is not None:
                            visualizer.plot_loss_accuracy(results_df, show=show_plots,
                                                         datasets=datasets_filter, metrics=metrics_filter)
                    case 'loss':
                        if results_df is not None:
                            visualizer.plot_loss(results_df, show=show_plots, datasets=datasets_filter)
                    case 'accuracy':
                        if results_df is not None:
                            visualizer.plot_accuracy(results_df, show=show_plots, datasets=datasets_filter)
                    case 'duration':
                        if profile_df is not None:
                            visualizer.plot_duration_table(profile_df, show=show_plots)
                    case 'timing':
                        if profile_df is not None:
                            visualizer.plot_epoch_timing(profile_df, show=show_plots)
                    case 'samples':
                        vis_samples(config, visualizer)
                    case 'model':
                        model_vis(config, visualizer)

            if show_plots:
                print("Showing plots...")
                plt.show()
            else:
                print(f"Visualizations saved to {output_dir}/")
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
