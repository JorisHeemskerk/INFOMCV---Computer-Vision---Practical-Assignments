# Assignment 2
> By Bas de Blok & Joris Heemskerk
---

## Getting started

1. Install Python version 3.14.2

Recommended method:
```
Conda create -n "INFOMCV_assignment_2" python=3.14.2
```
followed by
```
Conda activate INFOMCV_assignment_2
```
---
2. Within this Python, install the requirements

Recommended method:
```
pip install -r requirements.txt
```
---
3. If data was not provided, put data in the [data/](data/) folder. The structure of which matches the assignments'.

## Running code

### 1. Camera calibration.
For this, the primary script is [main.py](main.py). It can be run using:
```
python main.py
```
It outputs into config.xml's inside the data/ folder.\
Again, make sure to run this within the correct python environment.

### 2. Background subtraction
For this, use the [background_subtraction](background_subtraction.py) script. 
```
python background_subtraction.py
```
At the top of the script, configure which camera to calibrate. At the top, the already found optimal thresholds can be found and they can replace the lengthy calibration process. This file outputs two cache files, prints the optimal thresholds and creates a video of the background subtraction results.

### 3. Voxel reconstruction
For this, we used the provided [Computer-Vision-3D-Reconstruction
](https://github.com/dmetehan/Computer-Vision-3D-Reconstruction) GitHub package. Inside of [assignment.py](Computer-Vision-3D-Reconstruction/assignment.py), we implemented the voxel reconstruction. Run the code using [executable.py](Computer-Vision-3D-Reconstruction/executable.py), using:
```
python executable.py
``` 
This code outputs to an interactive window. These are the controls:
```
[WASD] - to move around
[G]    - to spawn in the next frame of the video
[C]    - To switch between colour mode and RED mode (applies to the next rendered frame)
[P]    - Render all the frames 1 by 1 in the video.
```

Make sure to execute this from the Computer-Vision-3D-Reconstruction directory.