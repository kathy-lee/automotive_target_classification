This is a object classification project based on radar spectrogram image. It aims to be a object classification sandkasten which is easy to use.

## About the dataset

This dataset is based on matlab simulation. See [here](https://www.mathworks.com/help/phased/examples/pedestrian-and-bicyclist-classification-using-deep-learning.html?s_eid=PEP_16543) to know about the scene description and simulation process. Generally, there are 5 scenes in the dataset: 
* 1 pedestrian 
* 1 bicyclist
* 1 pedestrian and 1 bicyclist 
* 2 pedestrians 
* 2 bicyclists

The spectrogram is generated from STFT transform of radar returns.

## Usage

The project is able to:
1. Visualization of data. Users can use cursor to browse spectrogram images of training/test dataset.
2. Object classification. Users can choose ML classification algorithm in sklearn and xgboost libraries or customize a NN(neural network) classification model in keras library with a configuration file easily. After classification, the misclassified scenes(spectrograms) can be shown with true label and misclassified label, user can browse these spectrograms with cursor. At the end of classification a .txt log file will be generated, which includes classification algorithm configuration parameter and classification performance evaluation. Logging brings better practice for hyperparameter tuning.

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
#### root_dir

the directory of training/test data.

#### (optional) sample_rate

the resampling rate of spectrogram image, on time axis and frequency axis. 

#### (optional) dimension_reduction

algorithm chosen from sklearn to reduce data dimension, and its customized parameters.

#### classifier: classification algorithm and its customized parameters.

1. For ML models, following classification algorithms are supported: 

* logistic regression
* svm
* decision tree
* knn
* random forest
* ada boost 
* gradient boost
* xgboost

The parameter keys which users would like to custormize should be given with the same parameter names in sklearn or xgboost libraries. 
    
2. For NN models, its parameters includes algorithm parameter and model parameter two parts.
The algorithm parameters 'optimizer'/'loss'/'metrics' should be given the same as in keras library. To extend the limited learning policies in kearas, following learning rate policies are supported:
* piecewise constant learning policy
* cyclic learning policy
* SGDR with warm start
* one-cycle learning policy
The learning policy parameter please refer to this link.

For NN model parameters, users can define sequential layerers in list form, each layer is a dictionary with customized parameters. The project now supports CNN and RNN type models.

### How to run

#### Data visualization

Run main.py with:

python main.py --show --datadir DATADIRECTORY --start STARTINDEX --type TRAINORTEST

#### Object classification

RUN main.py with:

python main.py --train --config CONFIGFILE --show-misclassified

### To be added

1. Transfer learning with pretrained model 
2. Automatically get optimal initial learning rate for NN model 




