import argparse
import datetime
import logging
import pytz
import torch
import traceback
import yaml

import numpy as np

from jsonschema import validate, ValidationError
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Any

from cat_dog_dataset import CatDogDataset
from create_logger import create_logger
from config.config_validation_template import CONFIG_TEMPLATE
from data import to_dataloaders
from train import train
from visualise import visualise_batch
from yolov1_base import YOLOv1Base
from yolov1_loss import YOLOv1Loss


def _process_job(
    job: dict[str, Any], 
    job_id: int, 
    dataset: Dataset, 
    logger: logging.Logger
)-> None:
    """
    This function executes the jobs according to their description.

    :param job: Job description, pulled from config
    :type job: dict[str, Any]
    :param job_id: ID of the current job (for logging).
    :type job_id: int
    :param dataset: Complete dataset object
    :type dataset: Dataset
    :param logger: Logger to log to.
    :type logger: logging.Logger
    """
    ####################################################################
    #                      Create the DataLoaders.                     #
    ####################################################################
    logger.debug(f"Splitting the dataset into {job["train_val_test_split"]}.")
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        job["train_val_test_split"]
    )
    logger.debug(
        f"{len(train_dataset)= }, {len(val_dataset)= }, {len(test_dataset)= }"
    )

    train_dataloader, val_dataloader, test_dataloader = to_dataloaders(
        [train_dataset, val_dataset, test_dataset], 
        batch_sizes=[job["batch_size"]] * 3, 
        shuffles=[True, True, False],
        logger=logger,
        # collate_fn=lambda x: tuple(zip(*x)) # TODO: why is this needed????????????
    )

    # Save quick example of the training dataloader to file.
    logger.debug("Visualising the first batch of the train dataloader.")
    visualise_batch(train_dataloader, "assignment_4/visualised_batch.png")

    ####################################################################
    #                          Load the model.                         #
    ####################################################################
    logger.debug("Initialising the model.")
    model = YOLOv1Base(logger)
    logger.debug(f"Model:\n{model}")
    logger.debug("Total number of parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    ####################################################################
    #                         Train the model.                         #
    ####################################################################
    N_EPOCHS = 5
    LEARNING_RATE = 0.001

    OPTIMISER = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    SCHEDULER = None
    LOSS_FN = YOLOv1Loss(job["lambda_coord"], job["lambda_noobj"])

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
            grid_size=CONFIG["general"]["grid_size"]
        )
    train_losses_std, train_accuracies_std = None, None
    val_losses_std, val_accuracies_std = None, None

    ####################################################################
    #                         Show the results.                        #
    ####################################################################
    print(
        f"\033[32mBest  training  accuracy: {max(train_accuracies)}, achieved "
        f"during epoch {np.argmax(train_accuracies) + 1}.\nBest validation "
        f"accuracy: {max(val_accuracies)}, achieved during epoch "
        f"{np.argmax(val_accuracies) + 1}.\033[37m"
    )

    ############## Visualise training accuracy and loss ################
    visualise_training(
        train_losses, 
        train_accuracies, 
        val_losses, 
        val_accuracies,
        train_losses_std, 
        train_accuracies_std,
        val_losses_std, 
        val_accuracies_std,
        model_name=model.__class__.__name__
    )

    test_loss, test_accuracy, test_labels, test_predictions = test_classes(
        dataloader=test_dataloader,
        model=model,
        loss_fn=LOSS_FN,
        device=DEVICE
    )
    print(
        f"\033[32mTest accuracy: {test_accuracy}, "
        f"test loss: {test_loss}\033[37m"
    )



def main()-> None:
    ####################################################################
    #                          Load the data.                          #
    ####################################################################
    dataset = CatDogDataset(
        img_dir=CONFIG["general"]["data_images_path"], 
        ann_dir=CONFIG["general"]["data_annotations_path"], 
        transform=transforms.Compose([
            transforms.Resize((
                CONFIG["general"]["input_image_size"],
                CONFIG["general"]["input_image_size"]
            )),
            transforms.ToTensor()
        ]),
        input_img_size=CONFIG["general"]["input_image_size"],
        grid_size=CONFIG["general"]["grid_size"]
    )

    ####################################################################
    #                         Execute all jobs.                        #
    ####################################################################
    for i, job in enumerate(CONFIG['jobs'].values()):
        logger.info(
           f"----- Processing Job {i:3.0f}/"
           f"{len(CONFIG['jobs'].values())-1:3.0f} -----"
        )
        logger.info(f"Job description: {job}")
        # This try-except catches individual job errors and attempts the 
        # next job if one of them crashes.
        try:
            if job in list(CONFIG['jobs'].values())[:i]:
                logger.warning(
                    "A job matching this exact configuration has already " 
                    "been executed. You likely have duplicate job descriptions"
                    ". This job will be skipped."
                )
                continue
            _process_job(
                job=job,
                job_id=i, 
                dataset=dataset,
                logger=logger
            )
        except KeyboardInterrupt as e:
            logger.critical(
                "PROGRAM MANUALLY HALTED BY KEYBOARD INTERRUPT "
                "(inside job execution loop)."
            )
            raise KeyboardInterrupt(
                "Keyboard interupt detected, halting program."
            ) from e
        except Exception as e:
            trace = ''.join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            logger.error(
                f"Error during handling of job {i} ({job = })\n\tTraceback:\n"
                f"\t{trace}\n\t'''{type(e)}: {e}'''\n"
                "Skipping this job, attempting to execute next job."
            )

if __name__ == "__main__":
    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description='configuration')
    parser.add_argument(
        '-c',
        '--config', 
        dest='config_file_path', 
        type=str, 
        default="assignment_4/config/config.yaml", 
        help="Path to config file. (default: %(default)s)"
    )
    args = parser.parse_args()

    # Initialise Logger.
    date = datetime.datetime.now(
        tz=pytz.timezone('Europe/Amsterdam')).strftime('%d-%m-%Y--%H-%M'
    )
    logger = create_logger(
        name="Computer Vision - Assignment 4", 
        output_log_file_name=f"assignment_4/logging/{date}.log"
    )
    logger.info(f"Provided commandline arguments: {args.__dict__}")

    # Seed PyTorch.
    torch.manual_seed(42)

    # Initialise Device.
    DEVICE = torch.accelerator.current_accelerator().type if \
        torch.accelerator.is_available() else "cpu"
    logger.info(f"Using {DEVICE} device")

    # validate the provided config file.
    with open(args.config_file_path, 'r') as stream:
        CONFIG = yaml.safe_load(stream)
    try:
        validate(CONFIG, CONFIG_TEMPLATE)
    except ValidationError as e:
        raise ValidationError(
            "\x1b[31;1mA validation error occurred in the config file" \
            f": {e.message}\x1b[0m"
        ) from e

    # Execute main
    main()
