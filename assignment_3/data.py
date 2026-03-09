import math
import matplotlib.pyplot as plt
import torch

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Any


def load_datasets(
    dataset: Callable, 
    root: str,
    train_val_partition: tuple[float, float],
    transform: Callable=ToTensor(),
    verbose: bool=True
)-> tuple[Dataset, Dataset, Dataset]:
    """
    Load a dataset directly as train, val, and test sets.

    :param dataset: Constructor for a pytorch dataset.
    :type datasets: Callable (specifically a Pytorch `datasets` object).
    :param root: Directory to save cached files to.
    :type root: str
    :param train_val_partition: Percentages for how much of the train 
        data should remain train and how much should be for validation.
    :type train_val_partition: tuple[float, float]
    :param verbose: Print info during process (DEFAULT=True)
    :type verbose: bool 
    :returns: Train, val, and test datasets, in that order.
    :rtype: tuple[Dataset, Dataset, Dataset]
    """
    
    assert sum(train_val_partition) == 1.0, "train_val_partition must sum to 1"

    training_data = dataset(
        root=root,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = dataset(
        root=root,
        train=False,
        download=True,
        transform=transform
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        training_data, train_val_partition
    )
    if verbose:
        print(
            f"Dataset sizes:\n\tTrain: {len(train_dataset)} datapoints"
            f"\n\tVal:   {len(val_dataset)} datapoints"
            f"\n\tTest: {len(test_dataset)} datapoints"
        )

    return train_dataset, val_dataset, test_dataset

def to_dataloaders(
    datasets: list[Dataset],
    batch_sizes: list[int],
    shuffles: list[bool],
    verbose: bool=True,
    **kwargs: dict[str, Any],
)-> list[DataLoader]:
    """
    Convert list of Dataset objects into DataLoaders.

    :param datasets: Datasets to convert.
    :type datasets: list[Dataset]
    :param batch_sizes: Batch size for the dataloaders.
    :type batch_sizes: list[int]
    :param shuffles: Shuffle the dataset order if True.
    :type shuffles: list[bool]
    :param verbose: Print info during process (DEFAULT=True)
    :type verbose: bool 
    :param **kwargs: Extra keyword arguments to pass to all dataloaders.
    :type **kwargs: dict
    :returns: List of converted datasets as DataLoader objects.
    :rtype: list[DataLoader]
    """
    dataLoaders = []
    assert len(datasets) == len(batch_sizes) == len(shuffles), \
        "One of the provided arguments has the wrong length: " \
        f"{len(datasets)=}, {len(batch_sizes)=}, {len(shuffles)=}"
    
    for dataset, batch_size, shuffle in zip(datasets, batch_sizes, shuffles):
        if verbose:
            print(
                f"Converting dataset of {len(dataset)} elements into "
                f"DataLoader with {len(dataset) // batch_size} partitions of "
                f"size {batch_size}.")
        dataLoaders.append(
            DataLoader(
                dataset=dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                **kwargs
            )
        )
    return dataLoaders

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
