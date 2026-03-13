import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from typing import Any


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
        model_name: str="Not specified"
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
    :param model_name: Name of the model used for top header title.
    :type model_name: str
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
    fig.suptitle(f"Model: {model_name}")
    plt.tight_layout()
    plt.savefig('assignment_3/results.png')
    plt.show()

def perform_tSNE(
    embeddings: np.ndarray, 
    labels: np.ndarray, 
    label_names: dict[int, str],
    images: np.ndarray | None=None,
    **kwargs: dict[str, Any]
)-> None:
    """
    Trains t-SNE model on embeddings and visiualises results.

    :param embeddings: embeddings for the entire dataset.
    :type embeddings: np.ndarray
    :param labels: list of integers corresponding to the datapoint  
        labels by index.
    :type labels: np.ndarray
    :param label_names: dictionary mapping the label indices to names.
    :type label_names: dict[int, str]
    :param images: The images to display on top of the dots if provided.
    :type images: np.ndarray | None
    :param **kwargs: Keyword arguments to pass to TSNE method.
    :type **kwargs: dict[str, Any]
    """
    print("\033[33mFitting t-SNE...\033[30m")
    tsne = TSNE(n_components=2, verbose=2, **kwargs)
    reduced = tsne.fit_transform(embeddings)
    print("\033[32mDone fitting t-SNE.\033[37m")

    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]

    fig, ax = plt.subplots(figsize=(20, 20))
    x, y = zip(*reduced)
    plt.scatter(x, y, c=colors, alpha=.5)

    ax.set_xlim(reduced[:, 0].min() - 5, reduced[:, 0].max() + 5)
    ax.set_ylim(reduced[:, 1].min() - 5, reduced[:, 1].max() + 5)

    handles = [
        plt.Line2D(
            [0], 
            [0], 
            marker='o', 
            color='w', 
            markerfacecolor=label_to_color[label],
            markeredgecolor=label_to_color[label],
            markersize=8, 
            label=label_names[label]
        ) for label in unique_labels
    ]
    plt.legend(
        handles=handles, 
        title="Classes", 
        loc='center left', 
        bbox_to_anchor=(1, 0.5)
    )

    plt.title("t-SNE projection of all datapoints")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.savefig('assignment_3/t-SNE_dots.png', bbox_inches='tight')

    if images is not None:
        for (x, y), image, color in zip(reduced, images, colors):
            imagebox = OffsetImage(image, zoom=.3)
            ab = AnnotationBbox(
                imagebox,
                (x, y),
                frameon=True,
                bboxprops=dict(
                    edgecolor=color,
                    linewidth=1.5,
                    boxstyle='square,pad=0.1'
                ),
                pad=0.1
            )
            ax.add_artist(ab)
        plt.savefig('assignment_3/t-SNE_images.png', bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: list[str]
)-> None:
    """
    Plot a confusion matrix.

    :param y_true: The true labels.
    :type y_true: np.ndarray
    :param y_pred: The predicted labels.
    :type y_pred: np.ndarray
    :param class_names: names of all classes.
    :type class_names: list[str]
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f'assignment_3/confusion_matrix_')
    plt.show()
