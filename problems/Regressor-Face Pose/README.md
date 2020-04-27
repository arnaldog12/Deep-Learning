# Problem Description

In this tutorial, I trained a model to predict roll, pitch, and yaw of a face presented to the camera. You can check the model running at CPU in the video below:

![](https://media.giphy.com/media/pPhEG3Grfxh4z2Hk6I/giphy.gif)

# Data
You can download the data used for this problem [here](https://drive.google.com/file/d/16FyZ6cOVFqvLoe1onR7Wn_XLl1ExbHWk/view?usp=sharing). Extract it in "data" folder.

# Folder Structure
Please, use the following folder structure to run all Jupyter Notebooks without any additional effort:

```
├── data (unzip dataset here)
├── images
├── models
├── Keras.ipynb
```

# Installation
I recommend using [Miniconda](https://conda.io/miniconda.html) to install all dependencies of this repository.
Since you have Miniconda installed, go to repository's folder and run the following commands:
```sh
$ conda env create -n face-pose numpy matplotlib keras scikit-learn jupyter
$ conda install -c conda-forge opencv dlib
```
To activate the environment in Linux/Mac:
```sh
$ source activate face-pose
```
To activate the environment in Windows:
```sh
$ activate face-pose
```
Finally, to run Jupyter Notebooks:
```sh
$ jupyter notebook
```
