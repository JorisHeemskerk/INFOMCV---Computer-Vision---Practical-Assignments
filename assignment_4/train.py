import copy
import logging
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable
from tqdm import tqdm

import handle_output

from early_stopper import EarlyStopper
from mean_average_precision import compute_map
from visualise import visualise_batch


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
    early_stopper: EarlyStopper,
    n_epochs: int,
    device: str,
    grid_size: int,
    iou_thresholds: list[float],
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
    train_losses_per_epoch, train_mAPs_per_epoch = [], []
    val_losses_per_epoch,   val_mAPs_per_epoch   = [], []
    for i in tqdm(range(n_epochs), "\033[33mEpoch"):
        print("\033[37m", end="") # Reset colour.
        logger.info(f"-----===== Epoch {i} (training) =====-----")
        train_loss_dict, train_mAPs = train_epoch(
            train_dataloader, 
            model, 
            loss_fn, 
            optimiser,
            device,
            grid_size,
            iou_thresholds,
            conf_threshold,
            logger
        )
        train_losses_per_epoch.append(train_loss_dict)
        train_mAPs_per_epoch.append(train_mAPs)

        logger.info(f"-----===== Epoch {i} (validation) =====-----")
        val_loss_dict, val_mAPs = val_epoch(
            val_dataloader, 
            model, 
            loss_fn, 
            device,
            grid_size,
            iou_thresholds,
            conf_threshold,
            logger
        )

        # TODO: what if the first threshold is not the best for this?
        if val_mAPs[str(iou_thresholds[0])] > (
            max(
                [d[str(iou_thresholds[0])] for d in val_mAPs_per_epoch]
            ) if len(val_mAPs_per_epoch) > 0 else -1
        ):
            best = copy.deepcopy(model.state_dict())
        val_losses_per_epoch.append(val_loss_dict)
        val_mAPs_per_epoch.append(val_mAPs)

        if scheduler is not None:
            scheduler.step(val_loss_dict["total"])

        if early_stopper.should_stop(val_loss_dict["total"]):
            logger.warning(f"Decided to stop early, at epoch {i}")
            break

    logger.info("Done training")
    model.load_state_dict(best)
    
    # Translate list of dicts to dict of lists.
    l_keys = train_losses_per_epoch[0].keys()
    mAP_keys = train_mAPs_per_epoch[0].keys()
    train_losses = {k: [d[k] for d in train_losses_per_epoch] for k in l_keys}
    train_mAPs = {k: [d[k] for d in train_mAPs_per_epoch] for k in mAP_keys}
    val_losses = {k: [d[k] for d in val_losses_per_epoch] for k in l_keys}
    val_mAPs= {k: [d[k] for d in val_mAPs_per_epoch] for k in mAP_keys}
    return train_losses, train_mAPs, val_losses, val_mAPs, model

def train_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    device: str,
    grid_size: int,
    iou_thresholds: list[float],
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
    train_losses = {
        "total": 0, 
        "xy": 0, 
        "wh": 0, 
        "conf_obj": 0, 
        "conf_noobj": 0, 
        "cls": 0
    }
    train_mAPs = {str(iou_threshold): [] for iou_threshold in iou_thresholds}

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
        
        for iou_threshold in train_mAPs.keys():
            train_mAPs[iou_threshold].append(
                compute_map(
                    y_hat, 
                    y, 
                    float(iou_threshold), 
                    conf_threshold
                ).item()
            )

        if batch % 100 == 0:
            train_loss, current = loss.item(), batch * len(y) + len(X)
            mAP_string = ", ".join(
                f"mAP@{threshold}: {np.mean(train_mAPs[threshold]):>2f}"
                for threshold in train_mAPs.keys()
            )
            logger.debug(
                f"train loss: {train_loss:>7f} | {mAP_string} | xy loss: "
                f"{loss_xy:>2f}, wh loss: {loss_wh:>2f}, conf loss: "
                f"{loss_conf_obj:>2f}, noobj conf loss: {loss_conf_noobj:>2f},"
                f" class loss: {loss_cls:>2f} | [{current:>5d}/"
                f"{len(dataloader.dataset):>5d}]"
            )
    
    return \
        {key: value / len(dataloader) for key, value in train_losses.items()},\
        {key: np.mean(value) for key, value in train_mAPs.items()}

def val_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str,
    grid_size: int,
    iou_thresholds: list[float],
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
    val_losses = {
            "total": 0, 
            "xy": 0, 
            "wh": 0, 
            "conf_obj": 0, 
            "conf_noobj": 0, 
            "cls": 0
        }
    val_mAPs = {str(iou_threshold): [] for iou_threshold in iou_thresholds}

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
            
            for iou_threshold in val_mAPs.keys():
                val_mAPs[iou_threshold].append(
                    compute_map(
                        y_hat, 
                        y, 
                        float(iou_threshold), 
                        conf_threshold
                    ).item()
                )
    avg_losses = {
        key: value / len(dataloader) for key, value in val_losses.items()
    }
    val_mAPs = {key: np.mean(value) for key, value in val_mAPs.items()}

    mAP_string = ", ".join(
        f"mAP@{threshold}: {np.mean(val_mAPs[threshold]):>2f}"
        for threshold in val_mAPs.keys()
    )
    logger.debug(
        f"Validation error | avg loss: {avg_losses["total"]:>7f} | "
        f"{mAP_string} | xy loss: {avg_losses["xy"]:>2f}, wh loss: "
        f"{avg_losses["wh"]:>2f}, conf loss: {avg_losses["conf_obj"]:>2f}, "
        F"noobj conf loss: {avg_losses["conf_noobj"]:>2f}, class loss: "
        f"{avg_losses["cls"]:>2f} |"
    )
    return avg_losses, val_mAPs

def test_classes(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str,
    grid_size: int,
    iou_thresholds: list[float],
    conf_threshold: float,
    plotting_conf_threshold: float,
    visualise_first_batch: bool,
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
    :param plotting_conf_threshold: Confidence for plotting the first
        batch, only used if `visualise_first_batch`.
    :type plotting_conf_threshold: float
    :param visualise_first_batch: True if you want to visualise the
        first batch of predictions along with the ground truth.
    :type visualise_first_batch: bool
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Average validation loss, accuracy, true labels, and 
        predicted labels.
    :rtype: tuple[float, float, np.ndarray, np.ndarray]
    """
    model.eval()
    test_losses = {
        "total": 0, 
        "xy": 0, 
        "wh": 0, 
        "conf_obj": 0, 
        "conf_noobj": 0, 
        "cls": 0
    }
    test_mAPs = {str(iou_threshold): [] for iou_threshold in iou_thresholds}

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            y_hat = y_hat.view(-1, grid_size, grid_size, 7)
            if i == 0 and visualise_first_batch == True:
                visualise_batch(
                    X, 
                    y, 
                    y_hat, 
                    plotting_conf_threshold,
                    dataloader.dataset.dataset.classes, 
                    f"{handle_output.OUTPUT_DIR}predict_batch_1.png"
                )

            loss, (
                loss_xy, loss_wh, loss_conf_obj, loss_conf_noobj, loss_cls
            ) = loss_fn(y_hat, y)
            
            test_losses["total"] += loss.item()
            test_losses["xy"] += loss_xy.item()
            test_losses["wh"] += loss_wh.item()
            test_losses["conf_obj"] += loss_conf_obj.item()
            test_losses["conf_noobj"] += loss_conf_noobj.item()
            test_losses["cls"] += loss_cls.item()
            
            for iou_threshold in test_mAPs.keys():
                test_mAPs[iou_threshold].append(
                    compute_map(
                        y_hat, 
                        y, 
                        float(iou_threshold), 
                        conf_threshold
                    ).item()
                )

    return \
        {key: value / len(dataloader) for key, value in test_losses.items()}, \
        {key: np.mean(value) for key, value in test_mAPs.items()}
