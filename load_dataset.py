from os.path import join as pjoin
from tensorflow.keras.utils import to_categorical
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

algo_map = {
    'pca': {"module": "sklearn.decomposition", "function": "PCA",
            "parameters": {"svd_solver": "randomized", "whiten": True}},
    'lda': {"module": "sklearn.discriminant_analysis", "function": "LinearDiscriminantAnalysis"},
    'ica': {"module": "sklearn.decomposition", "function": "FastICA"},
    'lr':  {"module": "sklearn.linear_model", "function": "LogisticRegression",
            "parameters": {"solver": 'lbfgs', "multi_class": 'multinomial', "max_iter": 500}},
    'svm': {"module": "sklearn.svm", "function": "SVC",
            "parameters": {"C": 10, "kernel": "rbf", "gamma": 0.1}},
    'decision tree': {"module": "sklearn.tree", "function": "DecisionTreeClassifier",
                      "parameters": {"max_depth": 40, "min_samples_split": 0.5, "min_samples_leaf": 1}},
    'knn': {"module": "sklearn.neighbors", "function": "KNeighborsClassifier",
            "parameters": {"n_neighbors": 5}},
    'random forest': {"module": "sklearn.ensemble", "function": "RandomForestClassifier",
                      "parameters": {"n_estimators": 100, "bootstrap": True, "max_samples": 0.5, "max_features": 0.5}},
    'ada boost': {"module": "sklearn.ensemble", "function": "AdaBoostClassifier",
                  "parameters": {"n_estimators": 100, "learning_rate": 0.1}},
    'gradient boost': {"module": "sklearn.ensemble", "function": "GradientBoostingClassifier",
                       "parameters": {"n_estimators": 200, "learning_rate": 0.1}},
    'xgboost': {"module": "xgboost", "function": "XGBClassifier",
                "parameters": {"learning_rate": 0.1,"n_estimators": 100, "max_depth": 5, "min_child_weight": 1,
                               "gamma": 0, "subsample": 0.8, "colsample_bytree": 0.8, "objective": 'binary:logistic',
                               "nthread": 4, "scale_pos_weight": 1, "seed": 27}},
    'cnn_a': {"module": "nnet_lib", "function": "cnn_a",
              "parameters": {"optimizer": "Adam", "learning_rate": 0.01, "loss": "categorical_crossentropy",
                           "metrics": "accuracy", "batch_size": 128, "epochs": 20}},
    'rnn_a': {"module": "nnet_lib", "function": "rnn_a",
              "parameters": {"optimizer": "Adam", "learning_rate": 0.01, "loss": "categorical_crossentropy",
                           "metrics": "accuracy", "batch_size": 128, "epochs": 20}}
}

def read_file(index, type, rootDir):
    file_index = "{0:02d}".format(index)
    if index == 0:
        file_name = type + 'NoCar' + '.h5'
    else:
        file_name = type + 'NoCar_' + file_index + '.h5'
    file_name = pjoin(rootDir, file_name)
    print(file_name)
    f = h5py.File(file_name, 'r')
    key = list(f.keys())[0]
    data = np.array(f[key])
    return data

def load_data(rootDir, sampRateT, sampRateF):
    print('load the data.')
    train_data = np.array([])
    for i in range(1,21):
        dataBlock = read_file(i, 'trainData', rootDir)
        dataBlock = dataBlock[:,:,0::sampRateT,0::sampRateF]
        train_data = np.vstack([train_data, dataBlock]) if train_data.size else dataBlock

    test_data = np.array([])
    for i in range(1,6):
        dataBlock = read_file(i, 'testData', rootDir)
        dataBlock = dataBlock[:, :, 0::sampRateT, 0::sampRateF]
        test_data = np.vstack([test_data, dataBlock]) if test_data.size else dataBlock

    train_data = np.transpose(train_data, (0, 3, 2, 1))
    test_data = np.transpose(test_data, (0, 3, 2, 1))

    # read label
    train_label = read_file(0, 'trainLabel', rootDir)
    train_label = train_label.flatten()
    train_label -= 1
    test_label = read_file(0, 'testLabel', rootDir)
    test_label = test_label.flatten()
    test_label -= 1

    print(train_data.shape)
    print(test_data.shape)

    print("Data sample distribution in training set: %d %d %d %d %d\n" % (np.count_nonzero(train_label == 1),
                                                                          np.count_nonzero(train_label == 2),
                                                                          np.count_nonzero(train_label == 3),
                                                                          np.count_nonzero(train_label == 4),
                                                                          np.count_nonzero(train_label == 0)))

    return train_data, train_label, test_data, test_label

def preprocess_data(train_data, train_label, test_data, test_label, classify_method):

    if classify_method.lower() in ['pca', 'lda', 'ica', 'lr', 'svm', 'decision tree', 'knn',
                                    'random forest', 'ada boost', 'gradient boost', 'xgboost']:
        print('preprocess data format for ML classifier:')
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)
    elif classify_method.lower() in ['cnn_a']:
        print('preprocess data format for CNN classifier:')
        train_label = to_categorical(train_label, num_classes=5)
        test_label = to_categorical(test_label, num_classes=5)
    elif classify_method.lower() in ['rnn_a']:
        print('preprocess data format for RNN classifier:')
        train_label = to_categorical(train_label, num_classes=5)
        test_label = to_categorical(test_label, num_classes=5)
        train_data = np.squeeze(train_data)
        test_data = np.squeeze(test_data)
        train_data = np.transpose(train_data, (0, 2, 1))
        test_data = np.transpose(test_data, (0, 2, 1))

    # scaler = StandardScaler()
    # train_data = scaler.fit_transform(train_data)
    # test_data = scaler.fit_transform(test_data)

    print(train_data.shape)
    print(test_data.shape)

    return train_data, train_label, test_data, test_label

def write_log(paramset, result):
    epoch_time = int(time.time())
    filename = "log" + str(epoch_time) + ".txt"
    with open(filename, "w+") as f:
        # write configuration parameters
        current_time = time.strftime("%d. %m. %Y. %H:%M:%S\n\n", time.localtime())
        f.write(current_time)
        f.write("resample in time axis: %d\n" % paramset["sample_rate"]["sample_rate_t"])
        f.write("resample in frequency axis: %d\n\n" % paramset["sample_rate"]["sample_rate_f"])
        if "dimension_redduction" in paramset:
            f.write("dimension reduction method: %s\n" % paramset["dimension_reduction"]["method"])
            f.write("dimension after reduction: %d\n" % paramset["dimension_reduction"]["n_components"])
            f.write("%s parameters: \n" % paramset["dimension_reduction"]["method"])
            for key, value in algo_map[paramset["dimension_reduction"]["method"]]["parameters"].items():
                f.write("\t%s : %s\n" % (key,str(value)))
        f.write("classifier method: %s\n" % paramset["classifier"]["method"])
        f.write("%s parameters: \n" % paramset["classifier"]["method"])
        for key, value in algo_map[paramset["classifier"]["method"]]["parameters"].items():
            f.write("\t%s : %s\n" % (key,str(value)))

        # write train/test results
        f.write("\ntraining performance: \n")
        f.write("confusion matrix: \n")
        train_conf = np.array_str(result["train_conf"])
        f.write(train_conf)
        train_precision = np.array_str(result["train_precision"])
        f.write("\naverage precision score: %s\n" % train_precision)
        train_recall = np.array_str(result["train_recall"])
        f.write("average recall score: %s\n" % train_recall)

        f.write("\ntest performance: \n")
        f.write("confusion matrix: \n")
        test_conf = np.array_str(result["test_conf"])
        f.write(test_conf)
        test_precision = np.array_str(result["test_precision"])
        f.write("\naverage precision score: %s\n" % test_precision)
        test_recall = np.array_str(result["test_recall"])
        f.write("average recall score: %s\n" % test_recall)
    return filename
