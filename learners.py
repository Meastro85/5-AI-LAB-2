import os
import math
import torch

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader


class Learner():

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def save(self, label):
        if not os.path.exists(os.path.join("checkpoints")):
            os.makedirs(os.path.join("checkpoints"))
        torch.save(self.model.state_dict(), os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label))

    def load(self, label):
        self.model.load_state_dict(torch.load(os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label), weights_only=True))
    

class RegressionLearner(Learner):

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__(model, criterion, optimizer)

    def train(self, dataloader: DataLoader, epochs: int, device: torch.device):

        self.model.to(device)

        train_losses = []

        for epoch in range(epochs):

            self.model.train()  # set model in training mode

            train_loss = 0

            for data, targets in dataloader:
                data = data.to(device)
                targets = targets.to(device)

                self.optimizer.zero_grad()
                predictions = self.model(data)
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(data)

            train_loss = train_loss / len(dataloader.dataset)

            if not train_losses or train_loss <= min(train_losses):
                self.save("leader")

            train_losses.append(train_loss)
            print(f"train epoch [{epoch + 1}/{epochs}]:\ttrain loss = {train_loss:.6f}")   

        print()
        
        self.save("final")

        return train_losses

    @torch.no_grad()
    def test(self, dataset: Dataset, device: torch.device):

        self.model.to(device)

        self.model.eval()  # set model in evaluation mode

        squared_error_sum = 0

        dataloader = DataLoader(dataset, batch_size=32)
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            predictions = self.model(data)
            squared_error_sum += torch.sum((predictions - targets) ** 2).item()
            mean_squared_error = squared_error_sum / len(dataset)

        return math.sqrt(mean_squared_error)

    @torch.no_grad()
    def predict(self, dataset, device):
        self.model.to(device)
        self.model.eval()
        data = dataset[:][0]
        data = data.to(device)
        results = self.model(data)
        return results


class ClassificationLearner(Learner):

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__(model, criterion, optimizer)

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int, device: torch.device):

        self.model.to(device)

        train_losses = []
        valid_losses = []
        valid_accuracies = []

        for epoch in range(epochs):
            
            self.model.train()  # set model in training mode

            train_loss = 0

            for data, targets in train_dataloader:
                data = data.to(device)
                targets = targets.to(device)
                
                self.optimizer.zero_grad()
                predictions = self.model(data)
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(data)

            train_loss = train_loss / len(train_dataloader.dataset)

            self.model.eval()  # set model in evaluation mode

            # evaluate the model on the validation dataset
            valid_loss, valid_accuracy = self._evaluate(valid_dataloader, device)    

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            print(f"train epoch [{epoch + 1}/{epochs}]:\ttrain loss = {train_loss:.4f}\tvalid loss = {valid_loss:.4f}\tvalid accuracy = {100 * valid_accuracy:.2f}%")   

            if len(valid_losses) == 1 or valid_loss <= min(valid_losses[:-1]):
                self.save("leader")

        print()
        
        self.save("final")

        return (train_losses, valid_losses, valid_accuracies)

    def test(self, test_dataloader: DataLoader, device: torch.device):

        self.model.to(device)

        self.model.eval()  # set model in evaluation mode

        # evaluate the model on the test dataset
        test_loss, test_accuracy = self._evaluate(test_dataloader, device)

        return (test_loss, test_accuracy)

    @torch.no_grad()
    def _evaluate(self, dataloader, device):
        eval_loss = 0
        correct_predictions = 0

        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            predictions = self.model(data)
            loss = self.criterion(predictions, targets)
            eval_loss += loss.item() * len(data)
            correct_predictions += (predictions.argmax(dim=1) == targets).sum().item()

        dataset_size = len(dataloader.dataset)
        eval_loss /= dataset_size
        eval_accuracy = correct_predictions / dataset_size

        return (eval_loss, eval_accuracy)
