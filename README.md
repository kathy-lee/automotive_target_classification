This is a object classification project based on radar spectrogram image.

## About the dataset

This dataset is based on matlab simulation. See [here](https://www.mathworks.com/help/phased/examples/pedestrian-and-bicyclist-classification-using-deep-learning.html?s_eid=PEP_16543) to know about the scene description and simulation process. Generally, there are 5 scenes in the dataset: 1 pedestrian, 1 bicyclist, 1 pedestrian and 1 bicyclist, 2 pedestrians, 2 bicyclists. The spectrogram is generated from STFT transform of radar returns.

## Usage

The project is able to:
1. Visualization of data. User can use cursor to browse spectrogram images of training/test dataset.
2. Object classification. User can choose any ML classification algorithm in sklearn or customize a NN(neural network) classification model in keras framework with a configuration file easily. After classification, the misclassified spectrograms can be shown with true label and misclassified label, user can browse these spectrograms with cursor. At the end of classification a .txt log file will be generated, which includes classification algorithm configuration parameter and classification performance evaluation. Logging brings better practice for hyperparameter tuning.

### Environment

* Ubuntu 18.04
* Python 3.7.7
* Tensorflow 2.0.0
* Keras 2.2.4-tf
* Sklearn 0.22.1

### Data preparation

1. Download data from [link](https://www.mathworks.com/supportfiles/SPT/data/PedBicCarData.zip).
2. Convert data and label files to `HDF5` format with *.m and *.m respectively.

### Configurations

A configuration file follows .json format and has the following parts:
1. root_dir: the directory of training/test data.
2. classifier: classification algorithm and its customized parameters.
    * For ML models, user can choose any algorithm in sklearn framework.
    * For NN models, its parameters includes algorithm parameter and model parameter two parts.
and can also have the following parts:
3. sample_rate: the resampling rate of spectrogram image, on time axis and frequency axis. 
4. dimension_reduction: algorithm chosen from sklearn to reduce data dimension, and its customized parameters.


### How to run


### To be added
1. Transfer learning with pretrained model 




