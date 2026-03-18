from ml_runner.core.config.schema import RunConfig
from main import apply_overrides

def test_run_config_from_dict_basic():
    cfg_dict = {
        "experiment": {"name": "test_exp"},
        "device": "cpu",
        "dataset": {"name": "MNIST", "params": {"root": "./data"}},
        "model": {"name": "MLP", "params": {"input_dim": 784}},
        "training": {"epochs": 12, "enabled": True},
        "evaluation": {"enabled": True, "eval_frequency": 2}
    }

    from ml_runner.core.extensions.extension_manager import ExtensionManager
    ext_manager = ExtensionManager()
    ext_cfg_classes = ext_manager.get_config_classes()

    run_cfg = RunConfig.from_dict(cfg_dict, extension_config_classes=ext_cfg_classes)


    assert run_cfg.experiment.name == "test_exp"
    assert run_cfg.device == "cpu"
    assert run_cfg.dataset.name == "MNIST"
    assert run_cfg.training.epochs == 12
    assert run_cfg.training.enabled is True
    assert run_cfg.evaluation.eval_frequency == 2

def test_apply_overrides():
    cfg_dict = {
        "optimizer": {"name": "adam", "lr": 0.001},
        "training": {"epochs": 5, "enabled": False}
    }

    class Args:
        lr = 0.01
        batch_size = 128
        epochs = None
        device = "cuda"
        checkpoint_dir = "my_runs"
        checkpoint_frequency = None
        do_train = True
        do_eval = None
        log_level = "DEBUG"
        log_dir = "logs"

    args = Args()
    overridden = apply_overrides(cfg_dict, args)

    assert overridden["optimizer"]["lr"] == 0.01
    assert overridden["dataloader"]["batch_size"] == 128
    assert overridden["training"]["epochs"] == 5 # Not overridden
    assert overridden["training"]["enabled"] is True
    assert overridden["device"] == "cuda"
    assert overridden["checkpoint"]["directory"] == "my_runs"
    assert overridden["logging"]["level"] == "DEBUG"
    assert overridden["logging"]["output_dir"] == "logs"

def test_run_config_with_overrides():
    cfg_dict = {
        "optimizer": {"name": "adam", "lr": 0.001},
        "training": {"epochs": 5}
    }

    class Args:
        lr = 0.01
        batch_size = 128
        epochs = 10
        device = "cpu"
        checkpoint_dir = "my_checkpoints"
        checkpoint_frequency = 5
        do_train = False
        do_eval = True
        log_level = None
        log_dir = None

    args = Args()
    from ml_runner.core.extensions.extension_manager import ExtensionManager
    ext_manager = ExtensionManager()
    ext_cfg_classes = ext_manager.get_config_classes()

    cfg_dict = apply_overrides(cfg_dict, args)
    run_cfg = RunConfig.from_dict(cfg_dict, extension_config_classes=ext_cfg_classes)

    assert run_cfg.optimizer.lr == 0.01
    assert run_cfg.dataloader["batch_size"] == 128
    assert run_cfg.training.epochs == 10
    assert run_cfg.training.enabled is False
    assert run_cfg.evaluation.enabled is True
    assert run_cfg.extensions["checkpoints"].directory == "my_checkpoints"
    assert run_cfg.extensions["checkpoints"].frequency == 5
