# Customizing Models and Datasets

`ml-runner` is built on a registry system that allows you to register your own PyTorch models and datasets without modifying the core engine.

## Registering a Custom Model

1. Create a Python file for your model.
2. Define your model as a subclass of `torch.nn.Module`.
3. Use the `@model_registry.register` decorator.

Example:

```python
import torch.nn as nn
from ml_runner.core.registries import model_registry

@model_registry.register("MyCustomModel")
class MyCustomModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)
```

4. **In your YAML config**:

   ```yaml
   model:
     name: MyCustomModel
     params:
       input_dim: 784
       hidden_dim: 128
       output_dim: 10
   ```

## Registering a Custom Dataset

Similarly, use the `@dataset_registry.register` decorator for datasets:

```python
from torch.utils.data import Dataset
from ml_runner.core.registries import dataset_registry

@dataset_registry.register("MyCustomDataset")
class MyCustomDataset(Dataset):
    def __init__(self, data_path: str):
        # Your data loading logic here
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        # Return a single data point
        return ...
```

4. **In your YAML config**:

   ```yaml
   dataset:
     name: MyCustomDataset
     params:
       data_path: "data/my_data.csv"
   ```

## Extensions

You can also create custom extensions to log metrics to other platforms or add custom training hooks. Check `src/ml_runner/extensions/` for examples of how the `TensorBoard` extension is implemented!
