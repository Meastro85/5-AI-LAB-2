import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from models import NeuralNetworkClassificationModel


RANDOM_SEED = 1
BATCH_SIZE = 64
LEARNING_RATES = [0.005, 0.01, 0.025, 0.05]
LEARNING_RATE_SEARCH_EPOCHS = 5
FINAL_EPOCHS = 50
VALIDATION_THRESHOLD = 0.9735
VALIDATION_BATCH_SIZE = 1000
OUTPUT_DIRECTORY = Path("experiment_outputs")


def make_loaders(dataset, batch_size):
    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, validation_dataset = random_split(
        dataset, [0.8, 0.2], generator=split_generator
    )
    train_loader_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, validation_loader


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct_count = 0
    sample_count = 0

    for data, targets in dataloader:
        data = data.to(device)
        targets = targets.to(device)
        logits = model(data)
        loss_sum += criterion(logits, targets).item() * len(data)
        correct_count += (logits.argmax(dim=1) == targets).sum().item()
        sample_count += len(data)

    return loss_sum / sample_count, correct_count / sample_count


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for data, targets in dataloader:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * len(data)
        sample_count += len(data)

    return loss_sum / sample_count


def train_for_epochs(model, train_loader, validation_loader, learning_rate, epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9
    )
    train_losses = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)
        print(
            f"epoch {epoch:>2}/{epochs}: "
            f"train loss={train_loss:.4f}, "
            f"validation loss={validation_loss:.4f}, "
            f"validation accuracy={100 * validation_accuracy:.2f}%"
        )

    return train_losses, validation_losses, validation_accuracies


def main():
    parser = argparse.ArgumentParser(
        description="Select an MNIST SGD learning rate, then run a 50-epoch experiment."
    )
    parser.add_argument(
        "--search-epochs",
        type=int,
        default=LEARNING_RATE_SEARCH_EPOCHS,
        help="Epochs for each learning-rate trial (default: 5).",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=LEARNING_RATES,
        help="Candidate learning rates to evaluate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=FINAL_EPOCHS,
        help="Epochs for the final training run (default: 50).",
    )
    args = parser.parse_args()
    if args.search_epochs <= 0 or args.epochs <= 0 or not args.learning_rates:
        parser.error("Epoch counts and the candidate learning-rate list must be positive.")
    if any(rate <= 0 for rate in args.learning_rates):
        parser.error("Learning rates must be positive.")

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}; training batch size: {BATCH_SIZE}")

    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rate_results = []
    rate_figure, rate_axis = plt.subplots(figsize=(9, 6))

    print("\nLearning-rate trials:")
    for learning_rate in args.learning_rates:
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
        model = NeuralNetworkClassificationModel().to(device)
        train_loader, validation_loader = make_loaders(dataset, BATCH_SIZE)
        train_losses, validation_losses, validation_accuracies = train_for_epochs(
            model,
            train_loader,
            validation_loader,
            learning_rate,
            args.search_epochs,
            device,
        )
        last_epoch_index = len(validation_losses) - 1
        result = {
            "learning_rate": learning_rate,
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "validation_accuracies": validation_accuracies,
            "best_validation_loss": validation_losses[last_epoch_index],
            "best_validation_accuracy": validation_accuracies[last_epoch_index],
            "best_epoch": last_epoch_index + 1,
        }
        rate_results.append(result)
        rate_axis.plot(
            range(1, args.search_epochs + 1),
            validation_losses,
            marker="o",
            label=f"lr={learning_rate:g}",
        )
        print(
            f"lr={learning_rate:g}: best validation loss="
            f"{result['best_validation_loss']:.4f}, "
            f"accuracy={100 * result['best_validation_accuracy']:.2f}% "
            f"at epoch {result['best_epoch']}"
        )

    rate_axis.set_xlabel("Epoch")
    rate_axis.set_ylabel("Validation cross-entropy loss")
    rate_axis.set_title(
        f"Learning-rate comparison (SGD, batch size {BATCH_SIZE})"
    )
    rate_axis.grid(True, alpha=0.3)
    rate_axis.legend()
    rate_figure.tight_layout()
    rate_figure.savefig(
        OUTPUT_DIRECTORY / "classification_learning_rate_comparison.png",
        dpi=150,
    )
    plt.show()

    selected = min(rate_results, key=lambda result: result["best_validation_loss"])
    learning_rate = selected["learning_rate"]
    print(
        f"\nSelected learning rate: {learning_rate:g} "
    )
    print(
        "Candidate rates: "
        + ", ".join(f"{rate:g}" for rate in args.learning_rates)
        + ". Selection uses validation loss"
    )

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    final_model = NeuralNetworkClassificationModel().to(device)
    train_loader, validation_loader = make_loaders(dataset, BATCH_SIZE)
    train_losses, validation_losses, validation_accuracies = train_for_epochs(
        final_model,
        train_loader,
        validation_loader,
        learning_rate,
        args.epochs,
        device,
    )

    first_threshold_epoch = next(
        (
            epoch
            for epoch, accuracy in enumerate(validation_accuracies, start=1)
            if accuracy >= VALIDATION_THRESHOLD
        ),
        None,
    )
    if first_threshold_epoch is None:
        print(
            f"\nValidation accuracy did not reach "
            f"{100 * VALIDATION_THRESHOLD:.2f}% within {args.epochs} epochs."
        )
    else:
        print(
            f"\nFirst epoch reaching at least "
            f"{100 * VALIDATION_THRESHOLD:.2f}% validation accuracy: "
            f"{first_threshold_epoch} "
            f"({100 * validation_accuracies[first_threshold_epoch - 1]:.2f}%)."
        )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, args.epochs + 1)
    axes[0].plot(epochs, train_losses, label="Training")
    axes[0].plot(epochs, validation_losses, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training and validation loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [100 * value for value in validation_accuracies])
    axes[1].axhline(
        100 * VALIDATION_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"Target {100 * VALIDATION_THRESHOLD:.2f}%",
    )
    if first_threshold_epoch is not None:
        axes[1].axvline(
            first_threshold_epoch,
            color="green",
            linestyle=":",
            label=f"First reached: epoch {first_threshold_epoch}",
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy (%)")
    axes[1].set_title("Validation accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY / "classification_50_epoch_training_curves.png",
        dpi=150,
    )
    plt.show()

    print(
        f"\n50-epoch summary with lr={learning_rate:g}, batch size={BATCH_SIZE}: "
        f"final train loss={train_losses[-1]:.4f}, "
        f"final validation loss={validation_losses[-1]:.4f}, "
        f"final validation accuracy={100 * validation_accuracies[-1]:.2f}%, "
        f"best validation accuracy={100 * max(validation_accuracies):.2f}%."
    )
    print(f"Plots saved in: {OUTPUT_DIRECTORY.resolve()}")
    print(
        "Discuss the learning curves together: decreasing training and "
        "validation loss indicates learning; a widening gap or rising "
        "validation loss while training loss falls suggests overfitting."
    )


if __name__ == "__main__":
    main()
