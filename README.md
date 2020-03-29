# Underwater-Buoy-Detection
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/urastogi885/underwater-buoy-detection/blob/master/LICENSE)

## Overview

Detection of buoys using gaussian mixture models (GMMs)

## Dependencies

- Python3
- Python3-tk
- Python3 Libraries: Numpy, OpenCV-Python

## Install Dependencies

- Install Python3, Python3-tk, and the necessary libraries: (if not already installed)

```
sudo apt install python3 python3-tk
sudo apt install python3-pip
pip3 install numpy opencv-python
```

- Check if your system successfully installed all the dependencies
- Open terminal using ```Ctrl+Alt+T``` and enter ```python3```.
- The terminal should now present a new area represented by ```>>>``` to enter python commands
- Now use the following commands to check libraries: (Exit python window using ```Ctrl + Z``` if an error pops up while running the below commands)

```
import tkinter
import numpy
import cv2
```

## Run

- Download video dataset from [here](https://drive.google.com/file/d/14VGYdseuSEVZD-AA4owDYFIY_53WfbrN/view)
- Using the terminal, clone this repository and go into the project directory, and run the main program:

```
https://github.com/urastogi885/underwater-buoy-detection
cd underwater-buoy-detection/Code
python3 buoy_detection.py
```

- An image window will pop-up, close this window, and press ```Ctrl + Z``` to stop the execution of the program.
