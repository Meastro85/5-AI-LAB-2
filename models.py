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

class NeuralNetworkClassificationModelCIFAR10(torch.nn.Module):
    def __init__(self):
            super().__init__()
            # TODO: Define the three classification layers.
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.maxPool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.maxPool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = torch.nn.Flatten()
            self.layer1 = torch.nn.Linear(4096, 128)
            self.dropout = torch.nn.Dropout()
            self.layer2 = torch.nn.Linear(128, 10)
            self.activation = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.maxPool1(x)
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.maxPool2(x)
        x = self.flatten(x)
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)