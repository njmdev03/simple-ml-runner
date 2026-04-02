
from ml_runner.core.config.schema import RunConfig
from ml_runner.core.extensions.extension_manager import ExtensionManager

def test_config_formats():
    print("--- Testing Nested Format ---")
    cfg_nested = {
        "model": {
            "MLP": {
                "input_dim": 784,
                "output_dim": 10
            }
        }
    }

    ext_manager = ExtensionManager()
    ext_cfg_classes = ext_manager.get_config_classes()

    run_cfg = RunConfig.from_dict(cfg_nested, extension_config_classes=ext_cfg_classes)
    print(f"Model Name: {run_cfg.model.name}")
    print(f"Model Params: {run_cfg.model.params}")
    assert run_cfg.model.name == "MLP"
    assert run_cfg.model.params["input_dim"] == 784

    print("\n--- Testing Old Format ---")
    cfg_old = {
        "model": {
            "name": "CNN",
            "channels": 32
        }
    }
    run_cfg_old = RunConfig.from_dict(cfg_old, extension_config_classes=ext_cfg_classes)
    print(f"Model Name: {run_cfg_old.model.name}")
    print(f"Model Params: {run_cfg_old.model.params}")
    assert run_cfg_old.model.name == "CNN"
    assert run_cfg_old.model.params["channels"] == 32

    print("\n--- Testing Mixed with params field ---")
    cfg_mixed = {
        "model": {
            "name": "Transformer",
            "params": {"n_heads": 8},
            "d_model": 512
        }
    }
    run_cfg_mixed = RunConfig.from_dict(cfg_mixed, extension_config_classes=ext_cfg_classes)
    print(f"Model Name: {run_cfg_mixed.model.name}")
    print(f"Model Params: {run_cfg_mixed.model.params}")
    assert run_cfg_mixed.model.name == "Transformer"
    assert run_cfg_mixed.model.params["n_heads"] == 8
    assert run_cfg_mixed.model.params["d_model"] == 512

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_config_formats()
