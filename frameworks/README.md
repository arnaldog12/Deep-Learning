This repo was inspired by [this article](https://medium.com/@tuennermann/convolutional-neural-networks-to-find-cars-43cbc4fb713) and gives a baseline source code to some of most known Deep Learning Frameworks: Tensorflow and Keras. Additionaly to Tensorflow, there are versions using Tensorboard and TFLearn. The same database and model have been used in order to allow code comparison between all implementations provided.

# Databases
I used both [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) datasets to train the models.

# Folder Structure
Please, use the following folder structure to run all Jupyter Notebooks without additional effort:

```
├── data (unzip datasets here)
├── log
├── models
├── Keras.ipynb
├── Tensorboard.ipynb
├── Tensorflow.ipynb
├── TFLearn.ipynb
└── deeplearning.yml
```

# Installation
I recommend using [Miniconda](https://conda.io/miniconda.html) to install all dependencies of this repository.
Since you have Miniconda installed, go to repository's folder and run the following command:
```sh
$ conda env create -f deeplearning.yml
```
To activate the environment in Linux/Mac:
```sh
$ source activate deeplearning
```
To activate the environment in Windows:
```sh
$ activate deeplearning
```
Finally, to run Jupyter Notebooks:
```sh
$ jupyter notebook
```
