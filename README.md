# Running Visual Speech Recognition using 3D-CNN

## Dataset: MIRACL-VC1
Link to the dataset: https://sites.google.com/site/achrafbenhamadou/-datasets/miracl-vc1. 

#### About the Dataset
* Fifteen speakers (10 women and 5 men)
* Ten words and ten phrases, each uttered 10 times
* Both depth and color images

## Pre-process Data
### Create videos
To extract lip regions, combine all color images into a video to feed into VisualizeLip.py using preprocess/make_videos.py file. This expects a command-line argument, the path to data.

> **Code/preprocess$** python make_videos.py VSR/data

### Convert video to images
The datatset will be a video so that it must be converted into a set of images. The models/VisualizeLip.py file does this. This file takes in 1 mandatory argument _**input**_ where *input* is the input path to video file (VSR/data). The _**output**_ path is hard-coded in such a way that it is compatible with the main script. Refer to the example below:

> **Code/models$** python VisualizeLip.py --input "VSR/data"

Make sure that the directory structure of VSR looks something like this (omitting **Phrases** from the **data**)

![Directory Structure](dir_struct.PNG)

## Predict output
Run the lip_reading.py file to predict the outputs of lip movements. Look at the example below:

> **Code$** python lip_reading.py
