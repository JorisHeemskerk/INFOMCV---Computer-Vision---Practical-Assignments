import torch
import numpy as np

from torch import nn
from torchvision import datasets
from torch.utils.data import ConcatDataset, Dataset

from train import train, train_cross_validation
from data import load_datasets, to_dataloaders
from lenet5_base import LeNet5Base


def finetune_cifar10(
    model: LeNet5Base,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    k_folds: int,
    device: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
)-> tuple[
    LeNet5Base,
    Dataset,
    torch.utils.data.dataloader.DataLoader,
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float]
]:
    """
    Finetune a model on the cifar 10 dataset.

    Change the outpt size of the given model to match the amount of
    classes in the cifar 10 dataset. Then train this model on the 
    cifar 10 dataset with the learning rate halved from what it was
    during pre-training.

    :param model: Model that will be finetuned
    :type model: LeNet5Base
    :param batch_size: batch size during training
    :type batch_size: int
    :param n_epochs: Number of epochs to train for.
    :type n_epochs: int
    :param learning_rate: Learnning rate by which the model trains.
    :type learning_rate: float
    :param k_folds: The number of folds to use.
    :type k_folds: int
    :param device: Device to move the model and data to.
    :type device: str
    :param scheduler: Scheduler to change how the learning rate adapts.
    :type scheduler: torch.optim.lr_scheduler.LRScheduler | None
    """
    train_dataset, val_dataset, test_dataset = load_datasets(
        dataset=datasets.CIFAR10, 
        root="assignment_3/data/", 
        train_val_partition=(.8, .2)
    )

    all_train_dataset = ConcatDataset([train_dataset, val_dataset])

    train_dataloader, val_dataloader, test_dataloader = \
        to_dataloaders(
            [train_dataset, val_dataset, test_dataset],
            batch_sizes=[batch_size] * 3,
            shuffles=[True, True, False]
        )
    
    N_CLASSES = len(test_dataset.classes)
    in_features = model.head[-1].in_features
    model.head[-1] = nn.Linear(in_features, N_CLASSES)
    nn.init.kaiming_uniform(model.head[-1].weight, nonlinearity="relu")
    nn.init.zeros_(model.head[-1].bias)

    model = model.to(device)

    OPTIMISER = torch.optim.Adam(params=model.parameters(), lr=learning_rate/2)
    if scheduler != None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            OPTIMISER, 
            step_size=5, 
            gamma=0.5
        )
    LOSS_FN = nn.CrossEntropyLoss()

    if k_folds is None:
        # Train it the normal way.
        train_losses, train_accuracies, val_losses, val_accuracies, model = \
            train(
                train_dataloader=train_dataloader, 
                val_dataloader=val_dataloader,
                model=model,
                loss_fn=LOSS_FN,
                optimiser=OPTIMISER,
                scheduler=scheduler,
                n_epochs=n_epochs,
                device=device   ,
            )
        train_losses_std, train_accuracies_std = None, None
        val_losses_std, val_accuracies_std = None, None
    else:
        # Use k-fold cross validation
        train_lossess, train_accuraciess, val_lossess, val_accuraciess, model=\
            train_cross_validation(
                full_train_dataset=all_train_dataset, 
                k_folds=k_folds,
                dataset_to_dataloader_function=lambda dataset: to_dataloaders(
                    [dataset],
                    batch_sizes=[batch_size],
                    shuffles=[False]
                ),
                model=model,
                loss_fn=LOSS_FN,
                optimiser=OPTIMISER,
                scheduler=scheduler,
                n_epochs=n_epochs,
                device=device,
            )
        train_losses = np.mean(train_lossess, axis=0)
        train_losses_std  = np.std(train_lossess, axis=0)

        train_accuracies = np.mean(train_accuraciess, axis=0)
        train_accuracies_std  = np.std(train_accuraciess, axis=0)
        
        val_losses = np.mean(val_lossess, axis=0)
        val_losses_std  = np.std(val_lossess, axis=0)
        
        val_accuracies = np.mean(val_accuraciess, axis=0)
        val_accuracies_std  = np.std(val_accuracies, axis=0)\
    
    return \
        model, \
        test_dataset, \
        test_dataloader, \
        train_losses, \
        train_accuracies, \
        val_losses, \
        val_accuracies, \
        train_losses_std, \
        train_accuracies_std, \
        val_losses_std, \
        val_accuracies_std
