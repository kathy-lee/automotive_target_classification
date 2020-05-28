from os.path import join as pjoin
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
            "parameters": {"solver": 'lbfgs', "multi_class": 'multinomial'}}
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
    train_data_raw = np.array([])
    for i in range(1,21):
        dataBlock = read_file(i, 'trainData', rootDir)
        dataBlock = dataBlock[:,:,0::sampRateT,0::sampRateF]
        train_data_raw = np.vstack([train_data_raw, dataBlock]) if train_data_raw.size else dataBlock

    print(train_data_raw.shape)
    #imgplot = plt.imshow(train_data_raw[7, 0, :, :])
    #plt.show()
    train_data = train_data_raw.reshape(train_data_raw.shape[0], -1)
    print(train_data.shape)

    test_data_raw = np.array([])
    for i in range(1,6):
        dataBlock = read_file(i, 'testData', rootDir)
        dataBlock = dataBlock[:, :, 0::sampRateF, 0::sampRateT]
        test_data_raw = np.vstack([test_data_raw, dataBlock]) if test_data_raw.size else dataBlock

    print(test_data_raw.shape)
    #imgplot = plt.imshow(test_data_raw[7, 0, :, :])
    #plt.show()
    test_data = test_data_raw.reshape(test_data_raw.shape[0], -1)
    print(test_data.shape)

    # read label
    train_label = read_file(0, 'trainLabel', rootDir)
    train_label = train_label.flatten()

    test_label = read_file(0, 'testLabel', rootDir)
    test_label = test_label.flatten()

    return train_data, train_label, test_data, test_label

def write_log(paramset, result):
    epoch_time = int(time.time())
    filename = "log" + str(epoch_time) + ".txt"
    with open(filename, "w+") as f:
        # write configuration parameters
        current_time = time.strftime("%d. %m. %Y. %H:%M:%S", time.localtime())
        f.write(current_time)
        f.write("resample in time axis: %d\n" % paramset["sample_rate"]["sample_rate_t"])
        f.write("resample in frequency axis: %d\n\n" % paramset["sample_rate"]["sample_rate_f"])
        f.write("dimension reduction method: %s\n" % paramset["dimension_reduction"]["method"])
        f.write("dimension after reduction: %d\n" % paramset["dimension_reduction"]["n_components"])
        f.write("%s parameters: \n" % paramset["dimension_reduction"]["method"])
        for key, value in algo_map[paramset["dimension_reduction"]["method"]]["parameters"].items():
            f.write("\t%s : %s\n" % (key,str(value)))
        f.write("classifier method: %s\n" % paramset["classifier"]["method"])
        f.write("num of iterations: %d\n" % paramset["classifier"]["n_iterations"])
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
