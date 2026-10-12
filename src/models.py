import torch
import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class DNNModel(nn.Module):
    """Matches the manuscript's stated architecture (Section 3.4): input ->
    128 -> ReLU -> 64 -> ReLU -> 1 -> sigmoid.
    """

    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def get_model(name, input_size):
    if name == "logreg":
        return LogisticRegressionModel(input_size)
    if name == "dnn":
        return DNNModel(input_size)
    raise ValueError(name)
