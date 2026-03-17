import os
import torch
import pickle

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Callable, Any


# superclass labels found on: https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR100_SUPERCLASSES = [
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium-sized_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2"
]


def load_datasets(
    dataset: Callable, 
    root: str,
    train_val_partition: tuple[float, float],
    train_transform: Callable=ToTensor(),
    eval_transform: Callable=ToTensor(),
    verbose: bool=True
)-> tuple[Subset, Subset, Dataset]:
    """
    Load a dataset directly as train, val, and test sets.

    :param dataset: Constructor for a pytorch dataset.
    :type datasets: Callable (specifically a Pytorch `datasets` object).
    :param root: Directory to save cached files to.
    :type root: str
    :param train_val_partition: Percentages for how much of the train 
        data should remain train and how much should be for validation.
    :type train_val_partition: tuple[float, float]
    :param train_transform: The transformations that are added onto the
        train set.
    :type train_transform: Callable
    :param eval_transform: The transformations that are added onto the
        val and test sets.
    :param verbose: Print info during process (DEFAULT=True)
    :type verbose: bool 
    :returns: Train, val, and test datasets, in that order.
    :rtype: tuple[Subset, Subset, Dataset]
    """
    
    assert sum(train_val_partition) == 1.0, "train_val_partition must sum to 1"

    training_data = dataset(
        root=root,
        train=True,
        download=True,
        transform=train_transform
    )
    evaluation_data = dataset(
        root=root,
        train=True,
        download=True,
        transform=eval_transform
    )
    test_dataset = dataset(
        root=root,
        train=False,
        download=True,
        transform=eval_transform
    )

    train_size = int(train_val_partition[0] * len(training_data))
    indices = torch.randperm(len(training_data))
    train_dataset = torch.utils.data.Subset(
        training_data,
        indices[:train_size]
    )
    val_dataset = torch.utils.data.Subset(
        evaluation_data,
        indices[train_size:]
    )
    if verbose:
        print(
            f"\033[30mDataset sizes:\n\tTrain: {len(train_dataset)} datapoints"
            f"\n\tVal:   {len(val_dataset)} datapoints"
            f"\n\tTest: {len(test_dataset)} datapoints\033[37m"
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
                f"\033[30mConverting dataset of {len(dataset)} elements into "
                f"DataLoader with {len(dataset) // batch_size} partitions of "
                f"size {batch_size}.\033[37m")
        dataLoaders.append(
            DataLoader(
                dataset=dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                **kwargs
            )
        )
    return dataLoaders

def cifar100_superclasses(
    root: str,
    train: Subset,
    val: Subset,
    test: Dataset
)-> None:
    """
    Set all target class labels to the superclass labels of the cifar100
    dataset. Afterwards cifar100 will use 20 labels instead of 100.

    :param root: Directory where cached files were saved to during 
        loading of the dataset.
    :type root: str
    :param train: Train dataset.
    :type train: Dataset
    :param val: Validation dataset.
    :type val: Dataset
    :param test: Test dataset.
    :type test: Dataset
    """
    for ds, filename in [
        (train, "train"),
        (val, "train"),
        (test, "test")
    ]:
        path = os.path.join(root, "cifar-100-python", filename)
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        if filename == "train":
            ds.dataset.targets = data[b'coarse_labels']
        else:
            ds.targets = data[b'coarse_labels']
    
    train.dataset.classes = CIFAR100_SUPERCLASSES
    val.dataset.classes = CIFAR100_SUPERCLASSES
    test.classes = CIFAR100_SUPERCLASSES
