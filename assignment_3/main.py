import numpy as np
import torch

from torch import nn
from torchvision import datasets
from torch.utils.data import ConcatDataset

from data import load_datasets, to_dataloaders
from train import train, train_cross_validation, embed_data
from lenet5_base import LeNet5Base
from lenet5_more_feature_kernels import LeNet5MoreFeatureKernels
from lenet5_extra_conv_layer import LeNet5ExtraConvLayer
from visualise import visualise_all_classes, visualise_training, perform_tSNE
from finetune import finetune_cifar10


torch.manual_seed(42)
DEVICE = torch.accelerator.current_accelerator().type if \
    torch.accelerator.is_available() else "cpu"
print(f"\033[1;35mUsing {DEVICE} device\033[0;37m")


def main()-> None:
    ####################################################################
    #                          Load the data.                          #
    ####################################################################
    DATASET = datasets.CIFAR100

    train_dataset, val_dataset, test_dataset = load_datasets(
        dataset=DATASET, 
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

    ####################################################################
    #                          Load the model.                         #
    ####################################################################
    N_CLASSES = len(test_dataset.classes)
    # model = LeNet5Base(n_classes=N_CLASSES)
    # model = LeNet5MoreFeatureKernels(
    #     n_classes=N_CLASSES, 
    #     n_first_layer_kernels=32
    # )
    model = LeNet5ExtraConvLayer(
        n_classes=N_CLASSES, 
        n_first_layer_kernels=32,
        n_channels=32
    )

    model.initialize_weights()
    model = model.to(DEVICE)
    ####################################################################
    #                     Set the hyperparemeters.                     #
    ####################################################################
    N_EPOCHS = 2
    LEARNING_RATE = 0.001
    K_FOLDS: int | None = None

    OPTIMISER = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    SCHEDULER = None
    # SCHEDULER = torch.optim.lr_scheduler.StepLR(
    #     OPTIMISER, 
    #     step_size=5, 
    #     gamma=0.5
    # )
    LOSS_FN = nn.CrossEntropyLoss()
    ####################################################################
    #                         Train the model.                         #
    ####################################################################
    if K_FOLDS is None:
        # Train it the normal way.
        train_losses, train_accuracies, val_losses, val_accuracies, model = \
            train(
                train_dataloader=train_dataloader, 
                val_dataloader=val_dataloader,
                model=model,
                loss_fn=LOSS_FN,
                optimiser=OPTIMISER,
                scheduler=SCHEDULER,
                n_epochs=N_EPOCHS,
                device=DEVICE,
            )
        train_losses_std, train_accuracies_std = None, None
        val_losses_std, val_accuracies_std = None, None
    else:
        # Use k-fold cross validation
        train_lossess, train_accuraciess, val_lossess, val_accuraciess, model=\
            train_cross_validation(
                full_train_dataset=all_train_dataset, 
                k_folds=5,
                dataset_to_dataloader_function=lambda dataset: to_dataloaders(
                    [dataset],
                    batch_sizes=[BATCH_SIZE],
                    shuffles=[False]
                ),
                model=model,
                loss_fn=LOSS_FN,
                optimiser=OPTIMISER,
                scheduler=SCHEDULER,
                n_epochs=N_EPOCHS,
                device=DEVICE,
            )
        train_losses = np.mean(train_lossess, axis=0)
        train_losses_std  = np.std(train_lossess, axis=0)

        train_accuracies = np.mean(train_accuraciess, axis=0)
        train_accuracies_std  = np.std(train_accuraciess, axis=0)
        
        val_losses = np.mean(val_lossess, axis=0)
        val_losses_std  = np.std(val_lossess, axis=0)
        
        val_accuracies = np.mean(val_accuraciess, axis=0)
        val_accuracies_std  = np.std(val_accuracies, axis=0)\

    if DATASET == datasets.CIFAR100:
        finetune_cifar10(
            model
        )

    print(
        f"\033[32mBest validation accuracy: {max(val_accuracies)}, achieved "
        f"during epoch {np.argmax(val_accuracies) + 1}.\033[37m"
    )
    
    model.save("assignment_3/model_cache")

    visualise_training(
        train_losses, 
        train_accuracies, 
        val_losses, 
        val_accuracies,
        train_losses_std, 
        train_accuracies_std,
        val_losses_std, 
        val_accuracies_std
    )
    ####################################################################
    #                   Perform t-SNE on test data.                    #
    ####################################################################
    # perform_tSNE(
    #     *embed_data(test_dataloader, model, DEVICE), 
    #     test_dataset.classes
    # )

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    end = time.time()
    print(f"Ran entire code in {end - start:.2f} seconds.")
