import itertools

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data import HousesDataset
from models import LinearRegressionModel
from utilities import plot_house_results


RANDOM_SEED = 1
TRAINING_DATASET_SIZE = 5000
EVALUATION_DATASET_SIZE = 1000
EPOCHS = 25
LEARNING_RATES = [0.001, 0.01, 0.05]
BATCH_SIZES = [32, 64]
DATASET_SIZES = [1000, 2000, 10000]


def dataset_prefix(dataset, size):
    subset = HousesDataset(0)
    subset.data = dataset.data[:size].clone()
    subset.targets = dataset.targets[:size].clone()
    subset.dataset_size = size
    return subset


def normalize_dataset(dataset, x_mean, x_std, y_mean, y_std):
    normalized = HousesDataset(0)
    normalized.data = (dataset.data - x_mean) / x_std
    normalized.targets = (dataset.targets - y_mean) / y_std
    normalized.dataset_size = len(dataset)
    return normalized


def fit_model(training_dataset, batch_size, learning_rate, epochs, device, seed):
    x_mean = training_dataset.data.mean(dim=0)
    x_std = training_dataset.data.std(dim=0).clamp_min(1e-8)
    y_mean = training_dataset.targets.mean(dim=0)
    y_std = training_dataset.targets.std(dim=0).clamp_min(1e-8)
    normalized_training_data = normalize_dataset(
        training_dataset, x_mean, x_std, y_mean, y_std
    )

    torch.manual_seed(seed)
    model = LinearRegressionModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    loader_generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        normalized_training_data,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for data, targets in training_loader:
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(data)
            loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Training diverged at epoch {epoch + 1}; "
                    f"learning_rate={learning_rate}, batch_size={batch_size}."
                )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(data)

        loss_history.append(epoch_loss / len(normalized_training_data))

    return model, (x_mean, x_std, y_mean, y_std), loss_history


@torch.no_grad()
def predict_prices(model, dataset, normalization, device):
    x_mean, x_std, y_mean, y_std = normalization
    data = ((dataset.data - x_mean) / x_std).to(device)
    normalized_predictions = model(data)
    return (normalized_predictions.cpu() * y_std + y_mean)


def evaluate_rmse(model, dataset, normalization, device):
    predictions = predict_prices(model, dataset, normalization, device)
    return torch.sqrt(torch.mean((predictions - dataset.targets) ** 2)).item()


def plot_dataset_performance(model, training_dataset, test_dataset, normalization, device, label):
    training_predictions = predict_prices(
        model, training_dataset, normalization, device
    )
    test_predictions = predict_prices(model, test_dataset, normalization, device)
    training_rmse = torch.sqrt(
        torch.mean((training_predictions - training_dataset.targets) ** 2)
    ).item()
    test_rmse = torch.sqrt(
        torch.mean((test_predictions - test_dataset.targets) ** 2)
    ).item()

    x_min = min(training_dataset.data.min().item(), test_dataset.data.min().item())
    x_max = max(training_dataset.data.max().item(), test_dataset.data.max().item())
    line_sizes = torch.linspace(x_min, x_max, 300).unsqueeze(1)
    line_prices = 5000 * line_sizes + 100000
    line_dataset = HousesDataset(0)
    line_dataset.data = line_sizes
    line_dataset.targets = line_prices
    line_predictions = predict_prices(model, line_dataset, normalization, device)

    plot_house_results(
        training_dataset.data,
        training_dataset.targets,
        training_predictions,
        line_sizes.squeeze(1),
        line_prices.squeeze(1),
        line_predictions.squeeze(1),
        f"Training set ({label}): RMSE = €{training_rmse:,.0f}",
    )
    plot_house_results(
        test_dataset.data,
        test_dataset.targets,
        test_predictions,
        line_sizes.squeeze(1),
        line_prices.squeeze(1),
        line_predictions.squeeze(1),
        f"Test set ({label}): RMSE = €{test_rmse:,.0f}",
    )
    return training_rmse, test_rmse


def main():
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(RANDOM_SEED)
    largest_training_dataset = HousesDataset(max(DATASET_SIZES))
    sweep_training_dataset = dataset_prefix(
        largest_training_dataset, TRAINING_DATASET_SIZE
    )
    size_sweep_datasets = {
        size: dataset_prefix(largest_training_dataset, size)
        for size in DATASET_SIZES
    }

    torch.manual_seed(RANDOM_SEED + 1)
    validation_dataset = HousesDataset(EVALUATION_DATASET_SIZE)
    torch.manual_seed(RANDOM_SEED + 2)
    test_dataset = HousesDataset(EVALUATION_DATASET_SIZE)

    sweep_results = []
    figure, axis = plt.subplots(figsize=(10, 6))

    for learning_rate, batch_size in itertools.product(LEARNING_RATES, BATCH_SIZES):
        model, normalization, losses = fit_model(
            sweep_training_dataset,
            batch_size,
            learning_rate,
            EPOCHS,
            device,
            RANDOM_SEED,
        )
        validation_rmse = evaluate_rmse(
            model, validation_dataset, normalization, device
        )
        sweep_results.append(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "model": model,
                "normalization": normalization,
                "losses": losses,
                "validation_rmse": validation_rmse,
            }
        )
        axis.plot(
            range(1, EPOCHS + 1),
            losses,
            label=f"lr={learning_rate:g}, batch={batch_size}",
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training MSE (standardized targets)")
    axis.set_title("Training loss: learning rate and batch size (5,000 samples)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()

    sweep_results.sort(key=lambda result: result["validation_rmse"])
    best = sweep_results[0]
    print("Hyperparameter comparison:")
    print("learning_rate  batch_size  final_train_MSE  validation_RMSE_euros")
    for result in sweep_results:
        print(
            f"{result['learning_rate']:>13g}  "
            f"{result['batch_size']:>10}  "
            f"{result['losses'][-1]:>15.6f}  "
            f"{result['validation_rmse']:>21.2f}"
        )
    print(
        "\nSelected configuration: "
        f"learning_rate={best['learning_rate']}, "
        f"batch_size={best['batch_size']}, "
        f"validation RMSE=€{best['validation_rmse']:.2f}"
    )
    print(
        "lowest validation RMSE -> most likely to generalize ."
    )

    final_model, final_normalization, _ = fit_model(
        sweep_training_dataset,
        best["batch_size"],
        best["learning_rate"],
        EPOCHS,
        device,
        RANDOM_SEED,
    )
    train_rmse, test_rmse = plot_dataset_performance(
        final_model,
        sweep_training_dataset,
        test_dataset,
        final_normalization,
        device,
        "5,000 training samples",
    )
    print(
        f"\nSelected model with 5,000 samples: "
        f"training RMSE=€{train_rmse:.2f}, test RMSE=€{test_rmse:.2f}"
    )

    print("\nEffect of training dataset size:")
    print("samples  training_RMSE_euros  test_RMSE_euros")
    for size, training_dataset in size_sweep_datasets.items():
        model, normalization, _ = fit_model(
            training_dataset,
            best["batch_size"],
            best["learning_rate"],
            EPOCHS,
            device,
            RANDOM_SEED,
        )
        training_rmse, size_test_rmse = plot_dataset_performance(
            model,
            training_dataset,
            test_dataset,
            normalization,
            device,
            f"{size:,} training samples",
        )
        print(f"{size:>7}  {training_rmse:>19.2f}  {size_test_rmse:>15.2f}")

    print(
        "\nlarger training sets usually provide a more stable estimate. "
        "Both scores vary due to noise in prices."
    )


if __name__ == "__main__":
    main()
