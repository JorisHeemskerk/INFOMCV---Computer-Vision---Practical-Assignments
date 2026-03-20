import copy
import logging
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable
from tqdm import tqdm
from mean_average_precision import compute_map


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
    grid_size: int,
    logger: logging.Logger
)-> tuple[
    dict[str, np.ndarray], 
    np.ndarray, 
    dict[str, np.ndarray], 
    np.ndarray, 
    nn.Module
]:
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
    :param grid_size: Size of the grid.
    :param grid_size: int
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Per epoch train losses, accuracies and validation losses 
        and accuracies. Along with model with the best val accuracy.
    :rtype: tuple[
        dict[str, np.ndarray], 
        np.ndarray, 
        dict[str, np.ndarray], 
        np.ndarray, 
        nn.Module
    ]
    """
    best = None
    best_val_accuracy = -1

    train_losses_per_fold: list[dict[str, list[float]]] = []
    val_losses_per_fold:   list[dict[str, list[float]]] = []
    train_accuracies_per_fold: list[list[float]] = []
    val_accuracies_per_fold:   list[list[float]] = []

    # Save the initial states to reset training every fold.
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimiser_state = copy.deepcopy(optimiser.state_dict())
    initial_scheduler_state = copy.deepcopy(
        scheduler.state_dict()
    ) if scheduler is not None else None

    fold_size = len(full_train_dataset) // k_folds
    for k in range(k_folds):
        logger.info(f"--==Fold {k+1}/{k_folds}==--")

        model.load_state_dict(copy.deepcopy(initial_model_state))
        optimiser.load_state_dict(copy.deepcopy(initial_optimiser_state))
        if scheduler is not None:
            scheduler.load_state_dict(copy.deepcopy(initial_scheduler_state))

        # Generate all the indexes of the items from each fold.
        val_idx = list(range(k * fold_size, k * fold_size + fold_size))
        train_idx = list(range(0, k * fold_size)) + \
            list(range(k * fold_size + fold_size, len(full_train_dataset)))

        train_dataloader = dataset_to_dataloader_function(
            Subset(full_train_dataset, train_idx)
        )[0]
        val_dataloader = dataset_to_dataloader_function(
            Subset(full_train_dataset, val_idx)
        )[0]

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
                grid_size=grid_size,
                logger=logger
            )
        

        fold_best_val_accuracy = max(val_accuracies)
        if fold_best_val_accuracy > best_val_accuracy:
            best_val_accuracy = fold_best_val_accuracy
            best = copy.deepcopy(model.state_dict())
        train_losses_per_fold.append(train_losses)
        train_accuracies_per_fold.append(train_accuracies)
        val_losses_per_fold.append(val_losses)
        val_accuracies_per_fold.append(val_accuracies)
    
    # Translate list of dicts of lists to dict-of-arrays, 
    # shape: (k_folds, n_epochs).
    loss_keys = train_losses_per_fold[0].keys()
    train_losses_stacked = {
        k: np.array([fold[k] for fold in train_losses_per_fold]) 
        for k in loss_keys
    }
    val_losses_stacked = {
        k: np.array([fold[k] for fold in val_losses_per_fold]) 
        for k in loss_keys
    }

    model.load_state_dict(best)
    return \
        train_losses_stacked, \
        np.array(train_accuracies_per_fold), \
        val_losses_stacked, \
        np.array(val_accuracies_per_fold), \
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
    grid_size: int,
    iou_threshold: float,
    conf_threshold: float,
    logger: logging.Logger
)-> tuple[
    dict[str, list[float]],
    list[float], 
    dict[str, list[float]], 
    list[float], nn.Module
]:
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
    :param grid_size: Size of the grid.
    :param grid_size: int
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Per epoch train losses, accuracies and validation losses 
        and accuracies. Along with model with the best val accuracy.
    :rtype: tuple[
        dict[str, list[float]],
        list[float], 
        dict[str, list[float]], 
        list[float], 
        nn.Module
    ]
    """
    best = None
    train_losses_per_epoch, train_mAPs = [], []
    val_losses_per_epoch,   val_mAPs   = [], []
    for _ in tqdm(range(n_epochs), "\033[33mEpoch"):
        print("\033[37m") # Reset colour.
        train_loss_dict, train_mAP = train_epoch(
            train_dataloader, 
            model, 
            loss_fn, 
            optimiser,
            device,
            grid_size,
            iou_threshold,
            conf_threshold,
            logger
        )
        train_losses_per_epoch.append(train_loss_dict)
        train_mAPs.append(train_mAP)

        val_loss_dict, val_mAP = val_epoch(
            val_dataloader, 
            model, 
            loss_fn, 
            device,
            grid_size,
            iou_threshold,
            conf_threshold,
            logger
        )

        if val_mAP > (
            max(val_mAPs) if len(val_mAPs) > 0 else -1
        ):
            best = copy.deepcopy(model.state_dict())
        val_losses_per_epoch.append(val_loss_dict)
        val_mAPs.append(val_mAP)

        if scheduler is not None:
            scheduler.step()
    
    logger.info("Done training")
    model.load_state_dict(best)
    
    # Translate list of dicts to dict of lists.
    keys = train_losses_per_epoch[0].keys()
    train_losses = {k: [d[k] for d in train_losses_per_epoch] for k in keys}
    val_losses   = {k: [d[k] for d in val_losses_per_epoch] for k in keys}
    return train_losses, train_mAPs, val_losses, val_mAPs, model

def train_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    device: str,
    grid_size: int,
    iou_threshold: float,
    conf_threshold: float,
    logger: logging.Logger
)-> tuple[dict[str, float], float]:
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
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Average training losses and accuracy over the epoch.
    :rtype: tuple[dict[str, float], float]
    """
    train_losses = {"total": 0, "xy": 0, "wh": 0, "conf_obj": 0, "conf_noobj": 0, "cls": 0}

    train_mAPs = []

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        y_hat = y_hat.view(-1, grid_size, grid_size, 7)
        loss, (loss_xy, loss_wh, loss_conf_obj, loss_conf_noobj, loss_cls) = \
            loss_fn(y_hat, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        train_losses["total"] += loss.item()
        train_losses["xy"] += loss_xy.item()
        train_losses["wh"] += loss_wh.item()
        train_losses["conf_obj"] += loss_conf_obj.item()
        train_losses["conf_noobj"] += loss_conf_noobj.item()
        train_losses["cls"] += loss_cls.item()
        
        train_mAPs.append(
            compute_map(y_hat, y, iou_threshold, conf_threshold).item()
        )

        if batch % 100 == 0:
            train_loss, current = loss.item(), batch * len(y) + len(X)
            logger.debug(
                f"train loss: {train_loss:>7f} | xy loss: {loss_xy:>2f}, "
                f"wh loss: {loss_wh:>2f}, conf loss: {loss_conf_obj:>2f}, "
                f"noobj conf loss: {loss_conf_noobj:>2f}, class loss: "
                f"{loss_cls:>2f} | [{current:>5d}/"
                f"{len(dataloader.dataset):>5d}]"
            )
    
    train_mAP = np.mean(train_mAPs)
    return \
        {key: value / len(dataloader) for key, value in train_losses.items()},\
        train_mAP

def val_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str,
    grid_size: int,
    iou_threshold: float,
    conf_threshold: float,
    logger: logging.Logger
)-> tuple[dict[str, float], float]:
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
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Average validation losses and accuracy over the epoch.
    :rtype: tuple[dict[str, float], float]
    """
    model.eval()
    val_losses = {"total": 0, "xy": 0, "wh": 0, "conf_obj": 0, "conf_noobj": 0, "cls": 0}

    val_mAPs = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            y_hat = y_hat.view(-1, grid_size, grid_size, 7)
            loss, (
                loss_xy, loss_wh, loss_conf_obj, loss_conf_noobj, loss_cls
            ) = loss_fn(y_hat, y)

            val_losses["total"] += loss.item()
            val_losses["xy"] += loss_xy.item()
            val_losses["wh"] += loss_wh.item()
            val_losses["conf_obj"] += loss_conf_obj.item()
            val_losses["conf_noobj"] += loss_conf_noobj.item()
            val_losses["cls"] += loss_cls.item()
            
            val_mAPs.append(
                compute_map(y_hat, y, iou_threshold, conf_threshold).item()
            )
    avg_losses = {
        key: value / len(dataloader) for key, value in val_losses.items()
    }
    val_mAP = np.mean(val_mAPs)
    logger.debug(
                f"Validation error | avg loss: {avg_losses["total"]:>7f} | xy "
                f"loss: {avg_losses["xy"]:>2f}, wh loss: {avg_losses["wh"]:>2f}"
                f", conf loss: {avg_losses["conf_obj"]:>2f}, noobj conf loss:"
                f" {avg_losses["conf_noobj"]:>2f}, class loss: "
                f"{avg_losses["cls"]:>2f} |"
            )
    return avg_losses, val_mAP

def test_classes(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str,
    grid_size: int,
    iou_threshold: float,
    conf_threshold: float,
    logger: logging.Logger
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
    test_losses = {"total": 0, "xy": 0, "wh": 0, "conf_obj": 0, "conf_noobj": 0, "cls": 0}

    test_mAPs = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            y_hat = y_hat.view(-1, grid_size, grid_size, 7)
            loss, (
                loss_xy, loss_wh, loss_conf_obj, loss_conf_noobj, loss_cls
            ) = loss_fn(y_hat, y)
            
            test_losses["total"] += loss.item()
            test_losses["xy"] += loss_xy.item()
            test_losses["wh"] += loss_wh.item()
            test_losses["conf_obj"] += loss_conf_obj.item()
            test_losses["conf_noobj"] += loss_conf_noobj.item()
            test_losses["cls"] += loss_cls.item()
            
            test_mAPs.append(
                compute_map(y_hat, y, iou_threshold, conf_threshold).item()
            )
    avg_losses = {
        key: value / len(dataloader) for key, value in test_losses.items()
    }
    test_mAP = np.mean(test_mAPs)

    return avg_losses, test_mAP
