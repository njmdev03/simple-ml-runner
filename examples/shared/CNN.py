import torch.nn as nn

class CNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # ----- Convolutional Feature Extractor -----
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # halves spatial dimension

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Adaptive pooling ensures compatibility with any image size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # ----- Fully Connected Classifier Head -----
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
