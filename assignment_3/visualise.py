import math
import matplotlib.pyplot as plt
import numpy as np


def visualise_all_classes(dataset: Dataset, labels: list[str])-> None:
    """
    Visualise 1 element of each class.

    :param dataset: A Pytorch Dataset object.
    :type datasets: Dataset
    :param labels: list of labels
    :type labels: list[str]
    """
    def best_grid(n):
        rows = round(math.sqrt(n))
        while n % rows != 0:
            rows -= 1
        cols = n // rows
        return rows, cols
    
    # Collect one sample per class.
    class_samples = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label not in class_samples:
            class_samples[label] = idx
        if len(class_samples) == len(labels):
            break

    figure = plt.figure(figsize=(8, 8))
    rows, cols = best_grid(len(labels))
    for i, class_idx in enumerate(sorted(class_samples.keys()), start=1):
        img, label = dataset[class_samples[class_idx]]
        figure.add_subplot(rows, cols, i)
        plt.title(labels[label])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()

def visualise_training(
        train_losses: list[float], 
        train_accuracies: list[float], 
        val_losses: list[float], 
        val_accuracies: list[float],
        train_losses_std: list[float] | None = None,
        train_accuracies_std: list[float] | None = None,
        val_losses_std: list[float] | None = None,
        val_accuracies_std: list[float] | None = None,
    )-> None:
    """
    Visualise both the loss and accuracy over the epochs, with optional
    shaded standard deviation bands.

    :param train_losses: Loss values during training.
    :type train_losses: list[float]
    :param train_accuracies: Accuracy values during training.
    :type train_accuracies: list[float]
    :param val_losses: Loss values during validation.
    :type val_losses: list[float]
    :param val_accuracies: Accuracy values during validation.
    :type val_accuracies: list[float]
    :param train_losses_std: Std of loss values during training. 
        (DEFAULT=None)
    :type train_losses_std: list[float] | None
    :param train_accuracies_std: Std of accuracy values during training. 
        (DEFAULT=None)
    :type train_accuracies_std: list[float] | None
    :param val_losses_std: Std of loss values during validation. 
        (DEFAULT=None)
    :type val_losses_std: list[float] | None
    :param val_accuracies_std: Std of accuracy values during validation. 
        (DEFAULT=None)
    :type val_accuracies_std: list[float] | None
    """
    fig, ax = plt.subplots(nrows=1, ncols=2)
    epochs = range(len(train_losses))

    def plot_with_band(axis, values, std, label):
        line, = axis.plot(epochs, values, label=label)
        if std is not None:
            values, std = np.array(values), np.array(std)
            axis.fill_between(
                epochs, 
                values - std, 
                values + std, 
                alpha=0.2, 
                color=line.get_color()
            )

    plot_with_band(ax[0], train_losses, train_losses_std, label="train loss")
    plot_with_band(ax[0], val_losses, val_losses_std, label="val loss")
    plot_with_band(
        ax[1], 
        train_accuracies, 
        train_accuracies_std, 
        label="train accuracy"
    )
    plot_with_band(
        ax[1], 
        val_accuracies, 
        val_accuracies_std, 
        label="val accuracy"
    )

    ax[0].set_title("Loss epochs")
    ax[1].set_title("Accuracy over epochs")
    ax[0].set_xlabel("Epochs")
    ax[1].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[1].set_ylabel("Accuracy")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('assignment_3/results.png')
    plt.show()
