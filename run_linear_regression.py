import torch

from data import HousesDataset

from models import LinearRegressionModel

from learners import RegressionLearner

from utilities import plot_house_sizes_and_prices, plot_house_results, plot_training_process


## PARAMETERS

random_seed = 1

batch_size = 64 # Batch size lager -> accuratere berekeningen maar trager

epochs = 1000 # Epochs hoger -> Meer training rondes, accuratere loss op het einde

learning_rate = 1e-4 # Adam: 10 stap size zegt tegen optimizer welke kan de loss reduced, Te groot: Loss wordt NaN, Te klein: Stabiel maar lange tijd nodig

train_dataset_size = 10000 # Meer training data -> meer resultaten om op te trainen
test_dataset_size = 20

## INITIALISATION

# manually set seeds for random number generators --> ensures reproducibility of results

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

## HARDWARE ACCELERATION

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('device:', device)
print()

# load the houses dataset

train_dataset = HousesDataset(train_dataset_size)
test_dataset = HousesDataset(test_dataset_size)

print("dataset dimensions:")
print()
print("train data shape:", tuple(train_dataset.data.shape))
print("test data shape:", tuple(test_dataset.data.shape))
print()

plot_house_sizes_and_prices(train_dataset[:][0], train_dataset[:][1], title="Train dataset")
plot_house_sizes_and_prices(test_dataset[:][0], test_dataset[:][1], title="Test dataset")

# create a dataloaders that sample batches from the train dataset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size)

## MODEL DEFINITION

"""START TODO: fill in the missing parts"""

# create an instance of the linear regression model

model = LinearRegressionModel()

# create an instance of an appropriate criterion (loss function) (see https://pytorch.org/docs/stable/nn.html#loss-functions)

criterion = torch.nn.MSELoss()

# create an instance of an appropriate opimizer to train the model parameters (see https://pytorch.org/docs/stable/optim.html)

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) #Adam

"""END TODO"""

print("linear model architecture:")
print()
print(model)
print()

learner = RegressionLearner(model, criterion, optimizer)

## TRAINING

# train the model
train_losses = learner.train(train_dataloader, epochs, device)

# plot the training loss versus the validation loss
plot_training_process(train_losses, [])

## TESTING

# load the best version of the model parameters
learner.load("leader")

# evaluate the model on the train dataset
train_loss = learner.test(train_dataset, device)
train_results = learner.predict(train_dataset, device).cpu()

print(f"performance on the train dataset:\troot mean squared error = {train_loss:.2f} €")
print()

# evaluate the model on the test dataset
test_loss = learner.test(test_dataset, device)
test_results = learner.predict(test_dataset, device).cpu()

print(f"performance on the test dataset:\troot mean squared error = {test_loss:.2f} €")
print()

# evaluate the model on the "unknown" linear function f(x)
linear_sizes = torch.linspace(train_dataset.min_house_size, train_dataset.max_house_size, 1000)
linear_prices = 5000 * linear_sizes + 100000
linear_dataset = (linear_sizes.unsqueeze(1), linear_prices.unsqueeze(1))
linear_results = learner.predict(linear_dataset, device).cpu()

plot_house_results(train_dataset.data, train_dataset.targets, train_results, linear_sizes, linear_prices, linear_results, title="Performance on train dataset")

plot_house_results(test_dataset.data, test_dataset.targets, test_results, linear_sizes, linear_prices, linear_results, title="Performance on test dataset")

print(f"unknown function: f(x) = 5000 * x + 100000 + {test_dataset.noise_house_price} * U(0, 1)")
print(f"learned function: g(x) = {model.linear_layer.weight.data[0].item()} * x + {model.linear_layer.bias.data[0].item()}")
print()