import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor

def plot_rgb_image_tensor(tensor: Tensor, title: str):
    """Plot an RGB image stored as an ``H x W x C`` floating-point tensor.

    ``H`` is the image height, ``W`` is the image width, and ``C`` must
    contain the three channels in RGB order. The channel values
    are expected to be in the range ``[0, 1]``.
    
    The tensor must not include a batch dimension.
    """
    arr = tensor.detach().cpu().numpy()
    plt.imshow(arr, interpolation="None")
    plt.title(title)
    plt.show()

def plot_grayscale_image_tensor(tensor: Tensor, title: str):
    """Plot a grayscale image stored as an ``H x W`` floating-point tensor.

    Values are expected to be in the range ``[0, 1]``, where ``0`` is black
    and ``1`` is white.
    
    The tensor must not include channel or batch dimensions.
    """
    arr = tensor.detach().cpu().numpy()
    plt.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="none")
    plt.title(title)
    plt.show()

def plot_scalar_image_tensor(tensor: Tensor, title: str):
    """Plot signed real scalars stored as an ``H x W`` floating-point tensor.

    Values may be negative or positive and do not need to use a fixed range.
    The color scale is centered on zero: negative values are blue, zero is
    white, and positive values are red.
    
    The tensor must not include channel or batch dimensions.
    """
    arr = tensor.detach().cpu().numpy()
    limit = np.abs(arr).max().item() or 1.0
    plt.imshow(arr, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="none")
    plt.colorbar(label="Value")
    plt.title(title)
    plt.show()


def plot_house_sizes_and_prices(sizes, prices, title):
    figure = plt.figure()
    ax = figure.add_subplot()
    ax.scatter(sizes, prices)
    ax.set_title(title)
    ax.set_xlabel("Size [m²]")
    ax.set_ylabel("Price [€]")
    figure.tight_layout()
    plt.show()


def plot_house_results(sizes, prices, estimations, linear_sizes, linear_prices, linear_estimations, title):
    figure = plt.figure()
    ax = figure.add_subplot()
    # plot real and estimated data points for dataset
    ax.scatter(sizes, prices, c="#377eb8")
    ax.scatter(sizes, estimations, c="#ff7f00")
    # plot real and estimated data points for a line
    ax.plot(linear_sizes, linear_prices, "#377eb8")
    ax.plot(linear_sizes, linear_estimations, "#ff7f00")
    ax.set_title(title)
    ax.set_xlabel("Size [m²]")
    ax.set_ylabel("Price [€]")
    ax.legend([
        "Actual house prices",
        "Estimated house prices",
        "Unknown function f(x)",
        "Learned function g(x)",
    ])
    plt.show()


def plot_images_and_labels(
    images,
    labels,
    predictions=None,
    class_names=None,
    normalization_mean=None,
    normalization_std=None,
):
    if (normalization_mean is None) != (normalization_std is None):
        raise ValueError("Provide both normalization_mean and normalization_std, or neither.")

    images = images[:20]
    labels = labels[:20]
    predictions = None if predictions is None else predictions[:20]
    figure = plt.figure()

    def format_label(value):
        value = value.item() if hasattr(value, "item") else value
        return class_names[int(value)] if class_names is not None else str(value)

    for i in range(len(images)):
        ax = figure.add_subplot(4, 5, i + 1)
        image = images[i].detach().cpu()
        if normalization_mean is not None:
            mean = image.new_tensor(normalization_mean).view(-1, 1, 1)
            std = image.new_tensor(normalization_std).view(-1, 1, 1)
            image = (image * std + mean).clamp(0, 1)

        if image.shape[0] == 1:
            ax.imshow(image[0], cmap="gray", interpolation="none")
        else:
            ax.imshow(image.permute(1, 2, 0), interpolation="none")

        label = format_label(labels[i])
        title = f"Label: {label}"
        if predictions is not None:
            title += f"\nPrediction: {format_label(predictions[i])}"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    figure.tight_layout()
    plt.show()


def plot_training_process(train_losses, valid_losses, valid_accuracies=None):
    epochs = np.arange(1, len(train_losses) + 1)

    if valid_accuracies is None:
        figure, ax = plt.subplots()
        ax.plot(epochs, train_losses, label="Training loss")
        if valid_losses:
            ax.plot(epochs, valid_losses, label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    else:
        figure, (loss_ax, accuracy_ax) = plt.subplots(2, 1, sharex=True)
        loss_ax.plot(epochs, train_losses, label="Training loss")
        loss_ax.plot(epochs, valid_losses, label="Validation loss")
        loss_ax.set_ylabel("Loss")
        loss_ax.legend()

        accuracy_ax.plot(epochs, np.asarray(valid_accuracies) * 100, label="Validation accuracy")
        accuracy_ax.set_xlabel("Epoch")
        accuracy_ax.set_ylabel("Accuracy (%)")
        accuracy_ax.set_ylim(0, 100)
        accuracy_ax.legend()

    figure.suptitle("Training process")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()