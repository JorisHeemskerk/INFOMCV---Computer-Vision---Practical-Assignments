import torch

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from typing import Callable


def load_datasets(
    dataset: Callable, 
    root: str,
    train_val_partition: tuple[float, float],
    transform: Callable=ToTensor(),
    verbose: bool=True
)-> tuple[Dataset, Dataset, Dataset]:
    
    assert sum(train_val_partition) == 1.0, "train_val_partition must sum to 1"

    training_data: Dataset = dataset(
        root=root,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset: Dataset = dataset(
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

# def to_dataloaders(
#     datasets: list[Dataset] | Dataset
# )-> list[DataLoader] | DataLoader:
#     if type(datasets) != list:
#         datasets = [datasets]
    