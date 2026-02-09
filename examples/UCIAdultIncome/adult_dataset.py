import os
import urllib.request
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class AdultDataset(Dataset):
    """PyTorch Dataset for UCI Adult Income with hardcoded categorical/numeric columns."""

    urls = {
        "train": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "test":  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    }

    # Ordered list of column names to use for the dataset.
    columns = [
        "age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race","sex",
        "capital-gain","capital-loss","hours-per-week","native-country","income"
    ]

    # Hardcoded column types. These are fixed and order doesn't matter.
    # Columns are split by type for transform and encoder computation.
    categorical_cols = [
        "workclass","education","marital-status","occupation",
        "relationship","race","sex","native-country"
    ]
    numeric_cols = [
        "age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"
    ]

    def __init__(self, root, train=True, download=False, transform=None,
                 encoders=None, target_encoder=None):
        self.train = train
        self.transform = transform
        os.makedirs(root, exist_ok=True)
        file_name = "adult.data" if train else "adult.test"
        self.path = os.path.join(root, file_name)

        if download and not os.path.exists(self.path):
            url = self.urls["train"] if train else self.urls["test"]
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, self.path)
            print("Download complete.")

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"{self.path} not found. Use download=True to fetch it.")

        # Load CSV
        df = pd.read_csv(
            self.path,
            names=self.columns,
            sep=r",\s+",
            engine="python",
            na_values="?"
        )

        # Remove first row in test set
        if not train:
            df = df.iloc[1:]

        # Drop missing rows
        df = df.dropna()

        # Strip strings
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Split features/target
        X = df.drop("income", axis=1)
        y = df["income"].str.replace(".", "", regex=False)

        # -------------------------------
        # Fit encoders on train
        # -------------------------------
        if train:
            self.encoders = {}
            for col in self.categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.encoders[col] = le

            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)

        # -------------------------------
        # Encode test set using train encoders
        # -------------------------------
        else:
            if encoders is None or target_encoder is None:
                raise ValueError("Test dataset requires train encoders and target_encoder.")

            self.encoders = encoders
            self.target_encoder = target_encoder

            # Encode only the known categorical columns
            for col in self.categorical_cols:
                unknown = set(X[col].unique()) - set(self.encoders[col].classes_)
                if unknown:
                    raise ValueError(f"Unknown categories in column '{col}': {unknown}")
                X[col] = self.encoders[col].transform(X[col])

            y = self.target_encoder.transform(y)

        # Convert all features to float tensor
        X = X.astype(np.float32)
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y
