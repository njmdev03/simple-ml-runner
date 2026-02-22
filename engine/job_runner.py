import torch
import os
import logging
from torch.utils.data import DataLoader
from string import Template
import pandas as pd

from engine.trainer import Trainer
from engine.evaluator import Evaluator
from reporting.exporter import Exporter
from reporting.profiler import Profiler

logger = logging.getLogger(__name__)

def get_device(priority_list):
    if isinstance(priority_list, str):
        priority_list = [d.strip() for d in priority_list.split(',')]
    for device in priority_list:
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if device == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")

def run_job(config):
    # Setup Profiler
    profiler = None
    if config.PROFILE:
        profiler = Profiler()
        profiler.start("total")

    # 2. Setup Device
    device = get_device(config.DEVICES)
    logger.info(f"Using device: {device}")

    # 3. Load Model
    if profiler: profiler.start("model_loading")
    model_val = config.MODEL
    if not model_val:
        logger.error(f"Error: MODEL not specified in config.")
        return

    model_obj = model_val

    if isinstance(model_obj, type):
        model = model_obj().to(device)
    else:
        model = model_obj.to(device)

    if profiler:
        model_dur = profiler.stop("model_loading")
        logger.info(f"Model loaded in {model_dur:.2f}s")

    # 4. Load Datasets
    if profiler: profiler.start("dataset_loading")

    # Conditionally load necessary datasets
    train_dataset = None
    if config.TRAIN or config.TEST_ON_TRAINING_DATA:
        train_dataset = config.TRAIN_DATASET

    test_dataset = None
    if config.TEST or config.TEST_WHILE_TRAINING:
        test_dataset = config.TEST_DATASET

    # Create Training Loader
    if config.TRAIN and train_dataset:
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Create Testing Loaders
    # Testing loader for test dataset
    if config.TEST:
        # Create Evaluator
        evaluator = Evaluator(config, model, device, profiler=profiler)

        if test_dataset:
            test_loader = DataLoader(test_dataset, batch_size=config.TESTING_BATCH_SIZE)

        # Testing loader for training dataset
        if config.TEST_ON_TRAINING_DATA and train_dataset:
            train_eval_loader = DataLoader(train_dataset, batch_size=config.TESTING_BATCH_SIZE)

    if profiler:
        ds_dur = profiler.stop("dataset_loading")
        logger.info(f"Datasets loaded in {ds_dur:.2f}s")

    # 5. Reporting Logic
    all_test_results = []
    skip_keys = set() # (epoch, dataset_name)

    save_path = config.SAVE_TESTS
    if save_path and os.path.exists(save_path):
        try:
            ext = os.path.splitext(save_path)[1].lower()
            if ext == '.csv':
                old_df = pd.read_csv(save_path)
            elif ext in ['.xlsx', '.xls']:
                old_df = pd.read_excel(save_path)
            else:
                old_df = None

            if old_df is not None:
                all_test_results = old_df.to_dict('records')
                # Explicitly populate skip_keys from old results
                for rec in all_test_results:
                    epoch = rec.get('epoch')
                    dataset = rec.get('dataset')
                    if epoch is not None and dataset:
                        skip_keys.add((int(epoch), str(dataset)))
                logger.info(f"Loaded {len(all_test_results)} existing test results. Work will be resumed.")
        except Exception as e:
            logger.warning(f"Failed to load existing results from {save_path}: {e}")

    # 6. Training
    try:
        if config.TRAIN and train_dataset:
            trainer = Trainer(config, model, device, profiler=profiler)

            def train_eval_cb(epoch):
                # Test on training data
                if train_eval_loader:
                    if (epoch, "Training") in skip_keys:
                        return
                    res = evaluator.evaluate(train_eval_loader, name=f"Epoch {epoch} Eval on Training")
                    res['epoch'] = epoch
                    cp_name_template = config.CHECK_MODEL_NAME
                    res['source'] = Template(cp_name_template).substitute(epoch=epoch) + ".pt"
                    res['dataset'] = "Training"
                    all_test_results.append(res)
                    skip_keys.add((epoch, "Training"))

                # Test on testing data
                if test_loader:
                    if (epoch, "Testing") in skip_keys:
                        return
                    res = evaluator.evaluate(test_loader, name=f"Epoch {epoch} Eval on Testing")
                    res['epoch'] = epoch
                    cp_name_template = config.CHECK_MODEL_NAME
                    res['source'] = Template(cp_name_template).substitute(epoch=epoch) + ".pt"
                    res['dataset'] = "Testing"
                    all_test_results.append(res)
                    skip_keys.add((epoch, "Testing"))

            trainer.run(train_loader, eval_callback=train_eval_cb if config.TEST_WHILE_TRAINING else None)

        # 7. Testing
        if config.TEST:
            if profiler: profiler.start("testing")

            if not config.TRAIN:
                evaluator = Evaluator(config, model, device, profiler=profiler)

            logger.info("--- Begin Evaluation ---")

            if config.TEST_CHECKPOINTS:
                if config.TEST_ON_TRAINING_DATA and train_eval_loader:
                    for res in evaluator.run_checkpoints(train_eval_loader, loader_name="Training", skip_keys=skip_keys):
                        all_test_results.append(res)
                        skip_keys.add((res['epoch'], "Training"))

                for res in evaluator.run_checkpoints(test_loader, loader_name="Testing", skip_keys=skip_keys):
                    all_test_results.append(res)
                    skip_keys.add((res['epoch'], "Testing"))

            logger.info("--- Final Model Evaluation ---")

            final_path = config.FINAL_OUTPUT_PATH
            if os.path.exists(final_path):
                # Final check for skip
                final_epoch = config.EPOCHS

                # Logic for skipping if already tested
                need_to_load = False
                if (final_epoch, "Testing") not in skip_keys:
                    need_to_load = True
                if config.TEST_ON_TRAINING_DATA and train_eval_loader and (final_epoch, "Training") not in skip_keys:
                    need_to_load = True

                if need_to_load:
                    model.load_state_dict(torch.load(final_path, map_location=device))

                    if config.TEST_ON_TRAINING_DATA and train_eval_loader and (final_epoch, "Training") not in skip_keys:
                        train_res = evaluator.evaluate(train_eval_loader, name="Training Data Final")
                        train_res['epoch'] = final_epoch
                        train_res['source'] = final_path
                        train_res['dataset'] = "Training"
                        all_test_results.append(train_res)
                        skip_keys.add((final_epoch, "Training"))

                    if (final_epoch, "Testing") not in skip_keys:
                        final_res = evaluator.evaluate(test_loader, name="Test Data Final")
                        final_res['epoch'] = final_epoch
                        final_res['source'] = final_path
                        final_res['dataset'] = "Testing"
                        all_test_results.append(final_res)
                        skip_keys.add((final_epoch, "Testing"))

            if profiler:
                test_duration = profiler.stop("testing")
                logger.info(f"Total testing time: {test_duration:.2f}s")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving partial results...")
    finally:
        # 8. Export
        if profiler:
            if profiler.is_active("total"):
                total_duration = profiler.stop("total")
                logger.info(f"Total process time: {total_duration:.2f}s")

            # Resolve Profile Output Path
            profile_path = config.PROFILE_OUTPUT
            if profile_path:
                Exporter.export([profiler.get_report()], profile_path)
                logger.info(f"Profiling results saved to {profile_path}")

        if save_path and all_test_results:
            Exporter.export(all_test_results, save_path)
