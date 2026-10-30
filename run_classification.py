import torch
import torchvision

from models import NeuralNetworkClassificationModel

from learners import ClassificationLearner

from utilities import plot_images_and_labels, plot_training_process


## PARAMETERS

random_seed = 1

batch_size_train = 32
batch_size_test = 1000

epochs = 5 #6

learning_rate = 0.025

## INITIALISATION

# manually set seeds for random number generators --> ensures reproducibility of results

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

## HARDWARE ACCELERATION

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("device:", device)
print()

## DATA

# load and transform the MNIST dataset

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

print("dataset dimensions:")
print()
print("train data shape:", tuple(train_dataset.data.shape))
print("train targets shape:", tuple(train_dataset.targets.shape))
print("test data shape:", tuple(test_dataset.data.shape))
print("test targets shape:", tuple(test_dataset.targets.shape))
print()

# split the training dataset in training and validation subsets
train_subset, valid_subset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])

# create dataloaders that sample batches from the training, validation and test datasets
train_dataloader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size_train, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_subset, batch_size=batch_size_test, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

# plot a sample of images and labels
sample_data, sample_targets = next(iter(test_dataloader))
plot_images_and_labels(sample_data, sample_targets)

## MODEL DEFINITION

"""START TODO: fill in the missing parts"""

# create an instance of the classifier model

model = NeuralNetworkClassificationModel()

# create an instance of an appropriate criterion (loss function) (see https://pytorch.org/docs/stable/nn.html#loss-functions)

criterion = torch.nn.CrossEntropyLoss()

# create an instance of an appropriate optimizer to train the model parameters (see https://pytorch.org/docs/stable/optim.html)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

"""END TODO"""

print("neural network architecture:")
print()
print(model)
print()

learner = ClassificationLearner(model, criterion, optimizer)

## TRAIN MODEL

# train the model
train_losses, valid_losses, valid_accuracies = learner.train(train_dataloader, valid_dataloader, epochs, device)

min_valid_loss = min(valid_losses)
min_valid_loss_index = valid_losses.index(min_valid_loss)
min_valid_accuracy = valid_accuracies[min_valid_loss_index]

print(f"train results:\tvalid loss = {min_valid_loss:.4f}\tvalid accuracy = {100 * min_valid_accuracy:.2f}%")   
print()

# plot the training loss versus the validation loss
plot_training_process(train_losses, valid_losses)

## TESTING

# load the best version of the model parameters
learner.load("leader")

# test the model
test_loss, test_accuracy = learner.test(test_dataloader, device)

print(f"test results:\ttest loss = {test_loss:.4f}\ttest accuracy = {100 * test_accuracy :.2f}%")
print()

# plot a sample of images, labels, and predictions
model.to(torch.device("cpu"))
sample_data, sample_targets = next(iter(test_dataloader))
sample_predictions = torch.argmax(model(sample_data), dim=1)
plot_images_and_labels(sample_data, sample_targets, sample_predictions)
