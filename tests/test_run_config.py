from config.run_config import RunConfig
from main import apply_overrides

def test_run_config_from_dict_basic():
    cfg_dict = {
        "experiment": {"name": "test_exp"},
        "device": "cpu",
        "dataset": {"MNIST": {"root": "./data"}},
        "model": {"MLP": {"input_dim": 784}},
        "training": {"epochs": 12, "batch_size": 32}
    }

    run_cfg = RunConfig.from_dict(cfg_dict)

    assert run_cfg.experiment.name == "test_exp"
    assert run_cfg.device == "cpu"
    assert run_cfg.dataset.name == "MNIST"
    assert run_cfg.dataset.params == {"root": "./data"}
    assert run_cfg.model.name == "MLP"
    assert run_cfg.model.params == {"input_dim": 784}
    assert run_cfg.training.epochs == 12

def test_apply_overrides():
    cfg_dict = {
        "optimizer": {"adam": {"lr": 0.001}},
        "training": {"epochs": 5}
    }

    # Mock args
    class Args:
        lr = 0.01
        batch_size = 128
        epochs = None
        device = "cuda"
        checkpoint_dir = None
        checkpoint_frequency = None

    args = Args()

    overridden = apply_overrides(cfg_dict, args)

    assert overridden["optimizer"]["adam"]["lr"] == 0.01
    assert overridden["dataloader"]["batch_size"] == 128
    assert overridden["training"]["epochs"] == 5 # Not overridden
    assert overridden["device"] == "cuda"

def test_run_config_with_overrides():
    cfg_dict = {
        "optimizer": {"adam": {"lr": 0.001}},
        "training": {"epochs": 5}
    }

    class Args:
        lr = 0.01
        batch_size = 128
        epochs = 10
        device = "cpu"
        checkpoint_dir = "my_checkpoints"
        checkpoint_frequency = 5

    args = Args()
    cfg_dict = apply_overrides(cfg_dict, args)
    run_cfg = RunConfig.from_dict(cfg_dict)

    assert run_cfg.optimizer.lr == 0.01
    assert run_cfg.dataloader["batch_size"] == 128
    assert run_cfg.training.epochs == 10
    assert run_cfg.checkpoint.directory == "my_checkpoints"
    assert run_cfg.checkpoint.frequency == 5
