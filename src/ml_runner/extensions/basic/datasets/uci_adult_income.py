import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from ml_runner.core.registries import Dataset


@Dataset("UCIAdult")
class UCIAdultDataset(TorchDataset):
    """
    UCI Adult Census Income Dataset
    https://archive.ics.uci.edu/ml/datasets/adult
    """
    def __init__(self, csv_path="./data/adult.csv", test_size=0.2, random_state=42):
        # Expect CSV with column 'income' as target
        self.df = pd.read_csv(csv_path)
        self.features = self.df.drop(columns=["income"])
        self.targets = self.df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

        X_train, X_test, y_train, y_test = train_test_split(
            self.features.values, self.targets.values, test_size=test_size, random_state=random_state
        )

        self.train_data = list(zip(torch.tensor(X_train, dtype=torch.float32),
                                   torch.tensor(y_train, dtype=torch.long)))
        self.test_data = list(zip(torch.tensor(X_test, dtype=torch.float32),
                                  torch.tensor(y_test, dtype=torch.long)))

    def get_split(self, train=True):
        return self.train_data if train else self.test_data

    def __len__(self):
        return len(self.train_data) + len(self.test_data)

    def __getitem__(self, idx):
        # By default, treat idx as index into train + test combined
        if idx < len(self.train_data):
            return self.train_data[idx]
        else:
            return self.test_data[idx - len(self.train_data)]
