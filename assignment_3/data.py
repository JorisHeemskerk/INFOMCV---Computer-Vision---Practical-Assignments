import torch

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Any


def load_datasets(
    dataset: Callable, 
    root: str,
    train_val_partition: tuple[float, float],
    train_tranform: Callable=ToTensor(),
    eval_transform: Callable=ToTensor(),
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
        transform=train_tranform
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
