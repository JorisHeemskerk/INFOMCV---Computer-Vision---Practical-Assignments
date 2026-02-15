# Assignment 1
> By Bas de Blok & Joris Heemskerk
---

## Getting started

1. Install Python version 3.14.2

Recommended method:
```
Conda create -n "INFOMCV_assignment_1" python=3.14.2
```
followed by
```
Conda activate INFOMCV_assignment_1
```
---
2. Within this Python, install the requirements

Recommended method:
```
pip install -r requirements.txt
```
---
3. If data was not provided, put data in the [data/](data/) folder. 
    - [data/all/](data/all/) should contain 20 images that are automatically detectable by OpenCV, along with 5 images that are not automatically detectable by OpenCV.
    - [data/auto/](data/auto/) should contain only 5 images that were automatically detectable.
    - [data/mix/](data/mix/) should contain 5 images that were automatically detectable (same 5 as above) along with 5 images that were not automatically detectable (same 5 from `all`).
    - [data/test/](data/test/) should contain 1 image that is not automatically detectable, with the chessboard at the edge of the screen.

If data has been provided, this is the test image:
![test image containing chess board](data/test/img_25.jpg)

If data has not been provided, it can be manually inserted into the data directories. Any naming issues can be resolved using the [rename_imgs.py](rename_imgs.py) script. This script will rename the images in chronological order.

## Running code
The primary script is [main.py](main.py). It can be run using:
```
python main.py
```
Again, make sure to run this within the correct python environment.

Inside the main.py script you will find instructions for which configurations you can run. You can follow the instructions there and comment in/out the code according to your needs.
