import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    save_final_dir: str | None=None
):
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
    :param n_epochs: Number of epochs to train for.
    :type n_epochs: int
    :param save_final_dir: Path to save final model to. (DEFAULT=None)
    :type save_final_dir: str | None
    """
    for _ in tqdm(range(n_epochs)):
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        val_epoch(val_dataloader, model, loss_fn)
    print("Done training")
    if save_final_dir is not None:
        filename = f"{model.__name__}.pth"
        print(f"Saving final model to {save_final_dir}/{filename}")
        torch.save(model, f"{save_final_dir}/{filename}")

def train_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer
)-> None:
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
    """
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(y) + len(X)
            print(
                f"loss: {loss:>7f}  "
                f"[{current:>5d}/{len(dataloader.dataset):>5d}]"
            )

def val_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module
)-> None:
    """
    Validate the accuracy and loss for a given dataset and model.

    :param dataloader: Dataset to validate with.
    :type dataloader: DataLoader
    :param model: Model to validate.
    :type model: nn.Module
    :param loss_fn: Loss function to validate with.
    :type loss_fn: nn.Module
    """
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            y_hat = model(X)
            test_loss += loss_fn(y_hat, y).item()
            correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: "
        f"{test_loss:>8f} \n"
    )
