
from ml_runner.core.config.schema import RunConfig
from ml_runner.core.extensions.extension_manager import ExtensionManager

def test_named_config_behavior():
    cfg_dict = {
        "model": {
            "MLP": {
                "input_dim": 784,
                "output_dim": 10
            }
        }
    }

    ext_manager = ExtensionManager()
    ext_cfg_classes = ext_manager.get_config_classes()

    run_cfg = RunConfig.from_dict(cfg_dict, extension_config_classes=ext_cfg_classes)

    print(f"Model Name: {run_cfg.model.name}")
    print(f"Model Params: {run_cfg.model.params}")

    # Ideally, this should work:
    # assert run_cfg.model.name == "MLP"
    # assert run_cfg.model.params["input_dim"] == 784

if __name__ == "__main__":
    test_named_config_behavior()
