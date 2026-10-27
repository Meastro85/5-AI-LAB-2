import torch

from torch import Tensor, Size
import datetime


class LinearModel():

    def __init__(self, input_shape: Size, output_shape: Size):
        
        if len(output_shape) != 1:
            raise ValueError("the output shape must be one-dimensional")
        
        self.weights = self.init_weights(input_shape, output_shape)

    def init_weights(self, input_shape: Size, output_shape: Size) -> Tensor:
        """Create the model's weight tensor."""
        # TODO: Initialize one small random weight for every input-output connection.
        shape = (*input_shape, *output_shape)

        rand_weights = 0.05 * torch.randn(shape, dtype=torch.float32)
        return rand_weights

    def calculate_output(self, input: Tensor) -> Tensor:
        """Apply the model to an input tensor."""
        # TODO: Calculate the unrestricted linear output without flattening the input.
        
        return torch.tensordot(input, self.weights, dims=input.dim())

    def calculate_loss(self, output: Tensor, target: Tensor) -> Tensor:
        """Measure the discrepancy between a model output and its target."""
        # TODO: Calculate the mean squared error.
        diff = output - target
        return torch.mean(diff**2)

    def calculate_gradients(self, input: Tensor, output: Tensor, target: Tensor) -> Tensor:
        """Calculate the gradient of the loss with respect to every model weight."""
        # TODO: Use the MSE and linear-layer derivatives to calculate d(loss)/d(weights) manually.
        dL_dy = 2.0 * (output - target) / output.numel()
        return torch.tensordot(input, dL_dy, dims=0)

    def update_weights(self, gradients: Tensor, learning_rate: float) -> Tensor:
        """Update the model parameters using the supplied gradients."""
        # TODO: Perform one gradient-descent step and return the updated weights.
        self.weights = self.weights - learning_rate * gradients
        return self.weights

    def train_step(self, input: Tensor, target: Tensor, learning_rate: float) -> float:
        """Run one optimization step for an input-target pair."""
        output = self.calculate_output(input)
        loss = self.calculate_loss(output, target)
        gradients = self.calculate_gradients(input, output, target)
        self.update_weights(gradients, learning_rate)

        return loss.item()

    def fit(self, input: Tensor, target: Tensor, steps: int, learning_rate: float) -> list[float]:
        """Optimize the model over multiple passes through an input-target pair."""
        loss_history = []

        for step in range(steps):
            loss = self.train_step(input, target, learning_rate)
            loss_history.append(loss)
            print(f"[{self.__class__.__name__}] fit step {step + 1}/{steps}: loss = {loss}")

        return loss_history


class LinearSigmoidModel(LinearModel):

    def calculate_output(self, input: Tensor) -> Tensor:
        """Apply the model to an input tensor."""
        # TODO: Calculate the linear logits and apply the sigmoid activation.
        logits = super().calculate_output(input)
        return 1.0 / (1.0 + torch.exp(-logits))

    def calculate_gradients(self, input: Tensor, output: Tensor, target: Tensor) -> Tensor:
        """Calculate the gradient of the loss with respect to every model weight."""
        # TODO: Use the MSE, sigmoid, and linear-layer derivatives to calculate d(loss)/d(weights) manually.
        dL_dy = 2.0 * (output - target) / output.numel()
        dL_dz = dL_dy * output * (1.0 - output)
        return torch.tensordot(input, dL_dz, dims=0)

