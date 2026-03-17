import datetime
import pytz
import torch

from torchvision import transforms

from cat_dog_dataset import CatDogDataset
from create_logger import create_logger
from data import to_dataloaders
from visualise import visualize_batch
from yolov1_base import YOLOv1Base


INPUT_IMAGE_SIZE = 112
date = datetime.datetime.now(
    tz=pytz.timezone('Europe/Amsterdam')).strftime('%d-%m-%Y--%H-%M'
)
logger = create_logger(
    name="Computer Vision - Assignment 4", 
    output_log_file_name=f"assignment_4/logging/{date}.log"
)
torch.manual_seed(42)
DEVICE = torch.accelerator.current_accelerator().type if \
    torch.accelerator.is_available() else "cpu"
logger.info(f"Using {DEVICE} device")


def main()-> None:
    ####################################################################
    #                          Load the data.                          #
    ####################################################################
    dataset = CatDogDataset(
        img_dir="assignment_4/data/images/", 
        ann_dir="assignment_4/data/annotations/", 
        transform=transforms.Compose([
            transforms.Resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)),
            transforms.ToTensor()
        ]),
        input_img_size=INPUT_IMAGE_SIZE
    )

    dataloader, = to_dataloaders(
        [dataset], 
        batch_sizes=[8], 
        shuffles=[True],
        logger=logger,
        # collate_fn=lambda x: tuple(zip(*x)) # TODO: why is this needed????????????
    )

    visualize_batch(dataloader)


    ####################################################################
    #                          Load the model.                         #
    ####################################################################
    model = YOLOv1Base(logger)
    logger.debug(f"Model:\n{model}")
    logger.debug("Total number of parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    ####################################################################
    #                     Set the hyperparemeters.                     #
    ####################################################################

    ####################################################################
    #                         Show the results.                        #
    ####################################################################
if __name__ == "__main__":
    main()
