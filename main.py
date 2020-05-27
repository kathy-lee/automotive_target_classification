import argparse
from load_dataset import load_data, write_log
import json
import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.decomposition import PCA
#from sklearn.decomposition import FastICA
import importlib
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

algo_map = {
    'pca': {"module": "sklearn.decomposition", "function": "PCA"},
    'lda': {"module": "sklearn.discriminant_analysis", "function": "LinearDiscriminantAnalysis"},
    'ica': {"module": "sklearn.decomposition", "function": "FastICA"},
    'lr':  {"module": "sklearn.linear_model", "function": "LogisticRegression"}
}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='pipeline configuration.', type=str)
    #parser.add_argument('--samplerate', default='2', help='resample image data to reduce data size', type=int)
    #parser.add_argument('--preprocess', default='pca', help='data dimensionality reduction', type=str)
    #parser.add_argument('--classifier', default='lr', help='classifier', type=str)
    args = parser.parse_args()
    params = vars(args)

    with open(params["config"], mode='r') as f:
        paramset = json.load(f)
    data_dir = paramset["root_dir"]
    samp_rate_t = paramset["sample_rate"]["sample_rate_t"]
    samp_rate_f = paramset["sample_rate"]["sample_rate_f"]
    dim_reducer = paramset["dimension_reduction"]["method"]
    num_components = paramset["dimension_reduction"]["n_components"]
    classify_method = paramset["classifier"]["method"]
    num_iter = paramset["classifier"]["n_iterations"]

    train_data, train_label, test_data, test_label = load_data(data_dir, samp_rate_t, samp_rate_f)

    print('\nbegin dimensionality reduction process.')
    module = importlib.import_module(algo_map[dim_reducer]["module"])
    reducer = getattr(module, algo_map[dim_reducer]["function"])(n_components=num_components)
    reducer.fit(train_data)
    train_feature = reducer.transform(train_data)
    print(train_feature.shape)
    test_feature = reducer.transform(test_data)
    print(test_feature.shape)

    print('\nbegin training process.')
    module = importlib.import_module(algo_map[classify_method]["module"])
    classifier = getattr(module, algo_map[classify_method]["function"])(solver='lbfgs', multi_class='multinomial', max_iter=num_iter)
    classifier.fit(train_feature, train_label)

    print('\npredict for test data.')
    test_pred = classifier.predict(test_feature)
    train_pred = classifier.predict(train_feature)

    print('\nevaluate the prediction(train data).')
    train_conf = confusion_matrix(train_label, train_pred)
    print(train_conf)
    train_precision = precision_score(train_label, train_pred, average=None)
    train_recall = recall_score(train_label, train_pred, average=None)
    print(train_precision)
    print(train_recall)

    print('\nevaluate the prediction(test data).')
    test_conf = confusion_matrix(test_label, test_pred)
    test_conf = np.array_str(test_conf)
    print(test_conf)
    test_precision = precision_score(test_label, test_pred, average=None)
    test_recall = recall_score(test_label, test_pred, average=None)
    print(test_precision)
    print(test_recall)

    pred_result = {
        "train_conf": train_conf,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "test_conf": test_conf,
        "test_precision": test_precision,
        "test_recall": test_recall
    }

    print('\nwrite log.')
    logFile = write_log(paramset, pred_result)
    print(logFile)

    # plt.plot(pca.explained_variance_ratio_)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.show()

if __name__ =="__main__":
    main()