import torch


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor):
        return self.linear_layer(x)


class NeuralNetworkClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define the three classification layers.
        self.flatten = torch.nn.Flatten()
        self.layer1 = torch.nn.Linear(28 * 28, 128)
        self.layer2 = torch.nn.Linear(128, 64)
        self.layer3 = torch.nn.Linear(64, 10)
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.layer3(x)
