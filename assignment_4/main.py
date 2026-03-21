import argparse
import logging
import numpy as np
import os
import shutil
import torch
import traceback
import yaml

from jsonschema import validate, ValidationError
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from typing import Any

import handle_output

from cat_dog_dataset import CatDogDataset
from create_logger import create_logger
from config.config_validation_template import CONFIG_TEMPLATE
from data import to_dataloaders
from early_stopper import EarlyStopper
from train import train, test_classes, train_cross_validation
from visualise import visualise_batch, visualise_training
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
    # Change output dir to specific job folder.
    handle_output.OUTPUT_DIR = f"{handle_output.OUTPUT_DIR}job_{job_id}/" if \
        job_id == 0 else "/".join(
            handle_output.OUTPUT_DIR.split("/")[:-2]
        ) + f"/job_{job_id}/"
    os.makedirs(handle_output.OUTPUT_DIR)
    ####################################################################
    #                      Create the DataLoaders.                     #
    ####################################################################
    logger.debug(f"Splitting the dataset into {job["train_val_test_split"]}.")
    labels = dataset._labels
    indices = list(range(len(dataset)))
    
    # Split in a stratisfied manner.
    train_idx, val_test_idx, _, val_test_labels = train_test_split(
        indices, 
        labels,
        test_size= \
            job["train_val_test_split"][1] + job["train_val_test_split"][2],
        stratify=labels,
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=job["train_val_test_split"][2] / (
            job["train_val_test_split"][1] + job["train_val_test_split"][2]
        ),
        stratify=val_test_labels,
        random_state=42
    )
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
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
    visualise_batch(
        train_dataloader, 
        job["plotting_conf_threshold"], 
        f"{handle_output.OUTPUT_DIR}train_batch_1_true.png"
    )

    ####################################################################
    #                          Load the model.                         #
    ####################################################################
    logger.debug("Initialising the model.")
    model = YOLOv1Base(logger)
    logger.debug(f"Model:\n{model}")
    logger.debug("Total number of parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    model = model.to(DEVICE)

    ####################################################################
    #                         Train the model.                         #
    ####################################################################
    OPTIMISER = torch.optim.Adam(
        params=model.parameters(),
        lr=job["learning_rate"],
        weight_decay=1e-4
    )
    SCHEDULER = None
    SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(
        OPTIMISER, mode='min', patience=10, factor=0.5
    )
    LOSS_FN = YOLOv1Loss(job["lambda_coord"], job["lambda_noobj"])
    EARLY_STOPPER = EarlyStopper(15, 0.0)

    # Arguments used by both normal training and cross_validation
    arguments = {
        "model" : model,
        "loss_fn" : LOSS_FN,
        "optimiser": OPTIMISER,
        "scheduler" : SCHEDULER,
        "early_stopper" : EARLY_STOPPER,
        "n_epochs" : job["n_epochs"],
        "device" : DEVICE,
        "grid_size" : CONFIG["general"]["grid_size"],
        "iou_thresholds" : job["iou_thresholds"],
        "conf_threshold" : job["conf_threshold"],
        "logger" : logger
    }
    # Only perform cross validation on k >= 2.
    if job["k_folds"] <= 1:
        train_losses, train_mAPs, val_losses, val_mAPs, model = train(
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
            **arguments
        )
        train_losses_std, train_mAPs_std = None, None
        val_losses_std, val_mAPs_std = None, None
    else:
        all_train_dataset = ConcatDataset([train_dataset, val_dataset])

        train_losses, train_mAPs, val_losses, val_mAPs, model = \
            train_cross_validation(
                full_train_dataset=all_train_dataset,
                k_folds=job["k_folds"],
                dataset_to_dataloader_function=lambda dataset: to_dataloaders(
                    [dataset],
                    batch_sizes=[job["batch_size"]],
                    shuffles=[False],
                    logger=logger
                ),
                **arguments, 
            )
        loss_keys = train_losses.keys()
        mAP_keys = train_mAPs.keys()
        train_losses_std = \
            {k: np.std(train_losses[k], axis=0) for k in loss_keys}
        train_losses = {k: np.mean(train_losses[k], axis=0) for k in loss_keys}
        val_losses_std = {k: np.std(val_losses[k], axis=0) for k in loss_keys}
        val_losses = {k: np.mean(val_losses[k], axis=0) for k in loss_keys}
        train_mAPs_std = {k: np.std(train_mAPs[k], axis=0) for k in mAP_keys}
        train_mAPs = {k: np.mean(train_mAPs[k], axis=0) for k in mAP_keys}
        val_mAPs_std = {k: np.std(val_mAPs[k], axis=0) for k in mAP_keys}
        val_mAPs = {k: np.mean(val_mAPs[k], axis=0) for k in mAP_keys}
    

    model.save(handle_output.OUTPUT_DIR)
    ####################################################################
    #                         Show the results.                        #
    ####################################################################
    # TODO: what if the first threshold is not the best for this?
    mAP_train_string = ", ".join(
        f"mAP@{threshold}: {np.max(train_mAPs[str(threshold)])*100:<2f}%"
        for threshold in job["iou_thresholds"]
    )
    train_best_epoch = np.argmax(train_mAPs[str(job["iou_thresholds"][0])]) + 1
    mAP_val_string = ", ".join(
        f"mAP@{threshold}: {np.max(val_mAPs[str(threshold)])*100:<2f}%"
        for threshold in job["iou_thresholds"]
    )
    val_best_epoch = np.argmax(val_mAPs[str(job["iou_thresholds"][0])]) + 1
    logger.critical(
        f"Best training scores: {mAP_train_string} | "
        f"achieved during epoch {train_best_epoch}."
    )
    logger.critical(
        f"Best validation scores: {mAP_val_string} | "
        f"achieved during epoch {val_best_epoch}."
    )

    visualise_training(
        train_losses, 
        train_mAPs, 
        val_losses, 
        val_mAPs, 
        handle_output.OUTPUT_DIR,
        train_losses_std, 
        train_mAPs_std, 
        val_losses_std, 
        val_mAPs_std
    )

    # run to visualise predictions on the first validation batch
    test_classes(
        dataloader=val_dataloader,
        model=model,
        loss_fn=LOSS_FN,
        device=DEVICE,
        grid_size=CONFIG["general"]["grid_size"],
        iou_thresholds=job["iou_thresholds"],
        conf_threshold=job["conf_threshold"],
        plotting_conf_threshold=job["plotting_conf_threshold"],
        visualise_first_batch=True,
        logger=logger
    )
    ####################################################################
    #                          Apply test set.                         #
    ####################################################################
    
    # TODO: comment in once final hyperparameters are selected

    # test_loss, test_mAP = test_classes(
    #     dataloader=test_dataloader,
    #     model=model,
    #     loss_fn=LOSS_FN,
    #     device=DEVICE,
    #     grid_size=CONFIG["general"]["grid_size"],
    #     iou_thresholds=job["iou_thresholds"],
    #     conf_threshold=job["conf_threshold"],
    #     logger=logger
    # )
    # print(
    #     f"\033[32mTest mAP: {test_mAP}, "
    #     f"Test error | avg loss: {test_loss["total"]:>7f} | xy "
    #     f"loss: {test_loss["xy"]:>2f}, wh loss: {test_loss["wh"]:>2f}"
    #     f", conf loss: {test_loss["conf_obj"]:>2f}, noobj conf loss:"
    #     f" {test_loss["conf_noobj"]:>2f}, class loss: "
    #     f"{test_loss["cls"]:>2f} |"
    # )



def main()-> None:
    ####################################################################
    #                          Load the data.                          #
    ####################################################################
    dataset = CatDogDataset(
        img_dir=CONFIG["general"]["data_images_path"], 
        ann_dir=CONFIG["general"]["data_annotations_path"], 
        input_img_size=CONFIG["general"]["input_image_size"],
        grid_size=CONFIG["general"]["grid_size"],
        logger=logger,
        transform=transforms.Compose([
            transforms.Resize((
                CONFIG["general"]["input_image_size"],
                CONFIG["general"]["input_image_size"]
            )),
            transforms.ToTensor()
        ]),
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
    os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)
    logger = create_logger(
        name="Computer Vision - Assignment 4", 
        output_log_file_name=f"{handle_output.OUTPUT_DIR}process.log"
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
    shutil.copy(args.config_file_path, handle_output.OUTPUT_DIR + "config.yml")

    ## Execute main. ###################################################
    main()
