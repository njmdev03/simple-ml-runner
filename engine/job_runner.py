import torch
import os
from torch.utils.data import DataLoader
from string import Template

from engine.trainer import Trainer
from engine.evaluator import Evaluator
from reporting.exporter import Exporter
from reporting.profiler import Profiler

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
    if not config.SILENT:
        print(f"Using device: {device}")

    # 3. Load Model
    if profiler: profiler.start("model_loading")
    model_val = config.MODEL
    if not model_val:
        print(f"Error: MODEL not specified in config.")
        return

    model_obj = model_val

    if isinstance(model_obj, type):
        model = model_obj().to(device)
    else:
        model = model_obj.to(device)
        
    if profiler:
        model_dur = profiler.stop("model_loading")
        if not config.SILENT:
            print(f"Model loaded in {model_dur:.2f}s")

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
        if not config.SILENT:
            print(f"Datasets loaded in {ds_dur:.2f}s")

    # 5. Training
    all_test_results = []
    tested_epochs = set()

    if config.TRAIN and train_dataset:
        trainer = Trainer(config, model, device, profiler=profiler)

        # Callback to run tests each epoch
        def train_eval_cb(epoch):
            # Test on training data
            if train_eval_loader:
                res = evaluator.evaluate(train_eval_loader, name=f"Epoch {epoch} Eval on Training")
                res['epoch'] = epoch
                cp_name_template = config.CHECK_MODEL_NAME
                res['source'] = Template(cp_name_template).substitute(epoch=epoch) + ".pt"
                res['dataset'] = "Training"
                all_test_results.append(res)

            # Test on testing data
            if test_loader:
                res = evaluator.evaluate(test_loader, name=f"Epoch {epoch} Eval on Testing")
                res['epoch'] = epoch
                cp_name_template = config.CHECK_MODEL_NAME
                res['source'] = Template(cp_name_template).substitute(epoch=epoch) + ".pt"
                res['dataset'] = "Testing"
                all_test_results.append(res)

            if train_eval_loader or test_loader:
                tested_epochs.add(epoch)

        trainer.run(train_loader, eval_callback=train_eval_cb if config.TEST_WHILE_TRAINING else None)

    # 6. Testing
    if config.TEST:
        if profiler: profiler.start("testing")

        # evaluator = Evaluator(config, model, device)

        # if test_dataset:
        #     test_loader = DataLoader(test_dataset, batch_size=config.get('TESTING_BATCH_SIZE') or config.get('BATCH_SIZE', 32))

        # if config.get('TEST_ON_TRAINING_DATA', False) and train_dataset:
        #     train_eval_loader = DataLoader(train_dataset, batch_size=config.get('BATCH_SIZE', 32))

        if not config.SILENT:
            print("--- Begin Evaluation ---")

        if config.TEST_CHECKPOINTS:
            if config.TEST_ON_TRAINING_DATA and train_eval_loader:
                checkpoint_results = evaluator.run_checkpoints(train_eval_loader, loader_name="Training", skip_epochs=tested_epochs)
                all_test_results.extend(checkpoint_results)

            checkpoint_results = evaluator.run_checkpoints(test_loader, loader_name="Testing", skip_epochs=tested_epochs)
            all_test_results.extend(checkpoint_results)

        if not config.SILENT:
            print("--- Final Model Evaluation ---")

        final_path = config.FINAL_OUTPUT_PATH
        if os.path.exists(final_path):
            model.load_state_dict(torch.load(final_path, map_location=device))

            if config.TEST_ON_TRAINING_DATA and train_eval_loader:
                train_res = evaluator.evaluate(train_eval_loader, name="Training Data Final")
                train_res['epoch'] = config.EPOCHS
                train_res['source'] = final_path
                train_res['dataset'] = "Training"
                all_test_results.append(train_res)

            final_res = evaluator.evaluate(test_loader, name="Test Data Final")
            final_res['epoch'] = config.EPOCHS
            final_res['source'] = final_path
            final_res['dataset'] = "Testing"
            all_test_results.append(final_res)

        if profiler:
            test_duration = profiler.stop("testing")
            if not config.SILENT:
                print(f"Total testing time: {test_duration:.2f}s")
                print()

    # 7. Export
    if profiler:
        total_duration = profiler.stop("total")
        if not config.SILENT:
            print(f"Total process time: {total_duration:.2f}s")
            print()

        # Resolve Profile Output Path
        profile_path = config.PROFILE_OUTPUT

        if profile_path:
            Exporter.export([profiler.get_report()], profile_path)
            if not config.SILENT:
                print(f"Profiling results saved to {profile_path}")

    save_path = config.SAVE_TESTS
    if save_path and all_test_results:
        Exporter.export(all_test_results, save_path)
