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
- Now use the following commands to check libraries: (Exit python window using ```Ctrl + Z``` if an error pops up while
running the below commands)

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
python3 buoy_detection.py input_video_location dataset_location output_destination
```

- Remember *input_video_location, dataset_location, output_destination* are file locations. The program takes these
locations relative to its own location.
- It is recommended that you provide locations within the project structure otherwise it will become too cumbersome for
you to reference the correct location of the file.
- Here is an example:

```
python3 buoy_detection.py videos/detectbuoy.avi images videos/video_output.avi
```

- The above file/folder locations are within the *Code* directory. Any other location within *Code* folder can be
referenced in the same way.
- Any location at the level of the *Code* folder can be referenced using *../file-name* for Ubuntu users. This is the
standard way of providing relative locations on linux terminals.
