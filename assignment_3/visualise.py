import math
import matplotlib.pyplot as plt

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
        val_accuracies: list[float]
    )-> None:
    """
    Visualise both the loss and accuracy over the epochs.

    :param train_losses: Loss values during training.
    :type train_losses: list[float]
    :param train_accuracies: Accuracy values during training.
    :type train_accuracies: list[float]
    :param val_losses: Loss values during validation.
    :type val_losses: list[float]
    :param val_accuracies: Accuracy values during validation.
    :type val_accuracies: list[float]
    """
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(train_losses, label="train loss")
    ax[1].plot(train_accuracies, label="train accuracy")
    ax[0].plot(val_losses, label="val loss")
    ax[1].plot(val_accuracies, label="val accuracy")
    ax[0].set_title("Loss epochs")
    ax[1].set_title("Accuracy over epochs")
    ax[0].set_xlabel("Epochs")
    ax[1].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[1].set_ylabel("Accuracy")
    ax[0].legend()
    ax[1].legend()
    plt.show()
    plt.tight_layout()
    plt.savefig('assignment_3/results.png')
