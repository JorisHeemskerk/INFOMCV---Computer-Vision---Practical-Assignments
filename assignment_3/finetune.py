from torch import nn
from torchvision import datasets
from torch.utils.data import ConcatDataset


from data import load_datasets, to_dataloaders

from lenet5_base import LeNet5Base


def finetune_cifar10(
    model: LeNet5Base,
):
    train_dataset, val_dataset, test_dataset = load_datasets(
        dataset=datasets.CIFAR10, 
        root="assignment_3/data/", 
        train_val_partition=(.8, .2)
    )

    all_train_dataset = ConcatDataset([train_dataset, val_dataset])
    # visualise_all_classes(train_dataset, test_dataset.classes)

    BATCH_SIZE = 32
    train_dataloader, val_dataloader, all_train_dataloader, test_dataloader = \
        to_dataloaders(
            [train_dataset, val_dataset, all_train_dataset, test_dataset],
            batch_sizes=[BATCH_SIZE] * 4,
            shuffles=[True, True, True, False]
        )
    
    N_CLASSES = len(test_dataset.classes)
    in_features = model.head[-1].in_features
    print(model)
    model.head[-1] = nn.Linear(in_features, N_CLASSES)
    print(model)

    