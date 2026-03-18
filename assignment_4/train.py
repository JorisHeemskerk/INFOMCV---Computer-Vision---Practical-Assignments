import copy
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable
from tqdm import tqdm


def train_cross_validation(
    full_train_dataset: Dataset, 
    k_folds: int,
    dataset_to_dataloader_function: Callable,
    model: nn.Module,
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    n_epochs: int,
    device: str,
)-> tuple[list[float], list[float], list[float], list[float]]:
    """
    Train a model for `n_epochs` epochs using k-fold cross validation.

    :param full_train_dataset: Dataset to train with.
    :type full_train_dataset: Dataset
    :param k_folds: The number of folds to use.
    :type k_folds: int
    :param model: Model to train.
    :type model: nn.Module
    :param loss_fn: Loss function to update gradients with.
    :type loss_fn: nn.Module
    :param optimiser: Optimiser used for backpropagation.
    :type optimiser: torch.optim.Optimizer
    :param scheduler: Scheduler to change how the learning rate adapts.
    :type scheduler: torch.optim.lr_scheduler.LRScheduler | None
    :param n_epochs: Number of epochs to train for.
    :type n_epochs: int
    :param device: Device to move data to.
    :type device: str
    :return: Per epoch train losses, accuracies and validation losses 
        and accuracies. Along with model with the best val accuracy.
    :rtype: tuple[
        np.ndarray, 
        np.ndarray, 
        np.ndarray, 
        np.ndarray, 
        nn.Module
    ]
    """
    best = None
    train_lossess, train_accuraciess, val_lossess, val_accuraciess = \
        [], [], [], []

    # Save the initial states to reset training every fold.
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimiser_state = copy.deepcopy(optimiser.state_dict())
    initial_scheduler_state = copy.deepcopy(
        scheduler.state_dict()
    ) if scheduler is not None else None

    fold_size = len(full_train_dataset) // k_folds
    for k in range(k_folds):
        print(f"\033[1;33m--==Fold {k+1}/{k_folds}==--\t\033[0;37m")

        model.load_state_dict(copy.deepcopy(initial_model_state))
        optimiser.load_state_dict(copy.deepcopy(initial_optimiser_state))
        if scheduler is not None:
            scheduler.load_state_dict(copy.deepcopy(initial_scheduler_state))

        # Generate all the indexes of the items from each fold.
        val_idx = list(range(k * fold_size, k * fold_size + fold_size))
        train_idx = list(range(0, k * fold_size)) + \
            list(range(k * fold_size + fold_size, len(full_train_dataset)))
        
        train_dataset = Subset(full_train_dataset, train_idx)
        val_dataset = Subset(full_train_dataset, val_idx)

        train_dataloader = dataset_to_dataloader_function(train_dataset)[0]
        val_dataloader = dataset_to_dataloader_function(val_dataset)[0]

        train_losses, train_accuracies, val_losses, val_accuracies, _ = \
            train(
                train_dataloader=train_dataloader, 
                val_dataloader=val_dataloader,
                model=model,
                loss_fn=loss_fn,
                optimiser=optimiser,
                scheduler=scheduler,
                n_epochs=n_epochs,
                device=device,
            )
        
        if (
            max(val_accuracies) if len(val_accuracies) > 0 else -1
        ) > \
        np.max(np.array(val_accuraciess)) if len(val_accuraciess) > 0 else -2:
            best = copy.deepcopy(model.state_dict())
        train_lossess.append(train_losses)
        train_accuraciess.append(train_accuracies)
        val_lossess.append(val_losses)
        val_accuraciess.append(val_accuracies)

    model.load_state_dict(best)
    return \
        np.array(train_lossess), \
        np.array(train_accuraciess), \
        np.array(val_lossess), \
        np.array(val_accuraciess), \
        model

def train(
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    n_epochs: int,
    device: str,
    grid_size: int
)-> tuple[list[float], list[float], list[float], list[float]]:
    """
    Train a model for `n_epochs` epochs.

    :param train_dataloader: Dataset to train with.
    :type train_dataloader: DataLoader
    :param val_dataloader: Dataset to validate with.
    :type val_dataloader: DataLoader
    :param model: Model to train.
    :type model: nn.Module
    :param loss_fn: Loss function to update gradients with.
    :type loss_fn: nn.Module
    :param optimiser: Optimiser used for backpropagation.
    :type optimiser: torch.optim.Optimizer
    :param scheduler: Scheduler to change how the learning rate adapts.
    :type scheduler: torch.optim.lr_scheduler.LRScheduler | None
    :param n_epochs: Number of epochs to train for.
    :type n_epochs: int
    :param device: Device to move data to.
    :type device: str
    :return: Per epoch train losses, accuracies and validation losses 
        and accuracies. Along with model with the best val accuracy.
    :rtype: tuple[
        list[float], 
        list[float], 
        list[float], 
        list[float], 
        nn.Module
    ]
    """
    best = None
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    for _ in tqdm(range(n_epochs), "\033[33mEpoch"):
        print("\033[37m") # Reset colour.
        train_loss, train_accuracy = train_epoch(
            train_dataloader, 
            model, 
            loss_fn, 
            optimiser,
            device,
            grid_size
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = val_epoch(
            val_dataloader, 
            model, 
            loss_fn, 
            device,
            grid_size
        )

        if val_accuracy > (
            max(val_accuracies) if len(val_accuracies) > 0 else -1
        ):
            best = copy.deepcopy(model.state_dict())
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        if scheduler is not None:
            scheduler.step()
    print("\033[1;32mDone training\033[0;37m")
    model.load_state_dict(best)
    return train_losses, train_accuracies, val_losses, val_accuracies, model

def train_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    device: str,
    grid_size: int
)-> tuple[float, float]:
    """
    Train a model for 1 epoch.

    :param dataloader: Dataset to train with.
    :type dataloader: DataLoader
    :param model: Model to train.
    :type model: nn.Module
    :param loss_fn: Loss function to update gradients with.
    :type loss_fn: nn.Module
    :param optimiser: Optimiser used for backpropagation.
    :type optimiser: torch.optim.Optimizer
    :param device: Device to move data to.
    :type device: str
    :param grid_size: Size of the grid the model used to divide the
        images.
    :type grid_size: int
    :return: Average training loss and accuracy over the epoch.
    :rtype: float
    """
    train_loss, correct = 0, 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        y_hat = y_hat.view(-1, grid_size, grid_size, 7)
        loss = loss_fn(y_hat, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        train_loss += loss.item()
        # correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()
        correct += 1

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(y) + len(X)
            print(
                f"\033[30mtrain loss: {loss:>7f}  "
                f"[{current:>5d}/{len(dataloader.dataset):>5d}]\033[37m"
            )
    return \
        train_loss / len(dataloader), \
        100 * (correct / len(dataloader.dataset))

def val_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str,
    grid_size: int
)-> tuple[float, float]:
    """
    Validate the accuracy and loss for a given dataset and model.

    :param dataloader: Dataset to validate with.
    :type dataloader: DataLoader
    :param model: Model to validate.
    :type model: nn.Module
    :param loss_fn: Loss function to validate with.
    :type loss_fn: nn.Module
    :param device: Device to move data to.
    :type device: str
    :return: Average validation loss and accuracy over the epoch.
    :rtype: float
    """
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            y_hat = y_hat.view(-1, grid_size, grid_size, 7)
            test_loss += loss_fn(y_hat, y).item()
            correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    print(
        f"\033[34mvalidation Error: \n Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n\033[37m"
    )
    return test_loss, 100 * correct

def test_classes(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Test the accuracy and loss for a given dataset and model.

    :param dataloader: Dataset to test with.
    :type dataloader: DataLoader
    :param model: Model to test.
    :type model: nn.Module
    :param loss_fn: Loss function to test with.
    :type loss_fn: nn.Module
    :param device: Device to move data to.
    :type device: str
    :return: Average validation loss, accuracy, true labels, and 
        predicted labels.
    :rtype: tuple[float, float, np.ndarray, np.ndarray]
    """
    model.eval()
    test_loss, correct = 0, 0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            test_loss += loss_fn(y_hat, y).item()
            
            predictions = y_hat.argmax(1)
            correct += (predictions == y).type(torch.float).sum().item()

            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return \
        test_loss / len(dataloader), \
        100 * (correct / len(dataloader.dataset)), \
        np.array(all_labels), \
        np.array(all_predictions)
