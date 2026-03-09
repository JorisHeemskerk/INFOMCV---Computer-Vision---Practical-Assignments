import torch

from torch import nn
from torchvision import datasets

from data import load_datasets, to_dataloaders, visualise_all_classes
from train import train

torch.manual_seed(42)
DEVICE = torch.accelerator.current_accelerator().type if \
    torch.accelerator.is_available() else "cpu"
print(f"Using {DEVICE} device")


def main()-> None:
    ####################################################################
    #                          Load the data.                          #
    ####################################################################
    train_data, val_data, test_data = load_datasets(
        dataset=datasets.CIFAR10, 
        root="assignment_3/data/", 
        train_val_partition=(.8, .2)
    )
    visualise_all_classes(train_data, test_data.classes)

    BATCH_SIZE = 32
    train_data, val_data, test_data = to_dataloaders(
        [train_data, val_data, test_data],
        batch_sizes=[BATCH_SIZE] * 3,
        shuffles=[True, True, False]
    )

    ####################################################################
    #                          Load the model.                         #
    ####################################################################
    lenet_5 = nn.Module()

    model = lenet_5
    ####################################################################
    #                     Set the hyperparemeters.                     #
    ####################################################################
    N_EPOCHS = 5
    LEARNING_RATE = 0.001

    OPTIMISER = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    LOSS_FN = nn.CrossEntropyLoss()
    ####################################################################
    #                         Train the model.                         #
    ####################################################################
    train(
        train_dataloader=train_data, 
        val_dataloader=val_data,
        model=model,
        loss_fn=LOSS_FN,
        optimizer=OPTIMISER,
        n_epochs=N_EPOCHS,
        device=DEVICE
    )

if __name__ == "__main__":
    main()
