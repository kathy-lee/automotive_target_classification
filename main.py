import argparse
from load_dataset import load_data, write_log, algo_map
import json
import numpy as np
import importlib
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

    train_data, train_label, test_data, test_label = load_data(data_dir, samp_rate_t, samp_rate_f)

    #scaler = MinMaxScaler()
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    if "dimension_redduction" in paramset:
        dim_reducer = paramset["dimension_reduction"]["method"]
        num_components = paramset["dimension_reduction"]["n_components"]
        algo_map[dim_reducer]["parameters"]["n_components"] = num_components
        print('\nbegin dimensionality reduction process.')
        module = importlib.import_module(algo_map[dim_reducer]["module"])
        reducer = getattr(module, algo_map[dim_reducer]["function"])(**algo_map[dim_reducer]["parameters"])
        reducer.fit(train_data)
        train_data = reducer.transform(train_data)
        print(train_data.shape)
        test_data = reducer.transform(test_data)
        print(test_data.shape)

    print('\nbegin training process.')
    classify_method = paramset["classifier"]["method"]
    module = importlib.import_module(algo_map[classify_method]["module"])
    classifier = getattr(module, algo_map[classify_method]["function"])(**algo_map[classify_method]["parameters"])
    classifier.fit(train_data, train_label)

    print('\npredict for test data.')
    test_pred = classifier.predict(test_data)
    train_pred = classifier.predict(train_data)

    print('\nevaluate the prediction(train data).')
    train_conf = confusion_matrix(train_label, train_pred)
    print(train_conf)
    train_precision = precision_score(train_label, train_pred, average=None)
    train_recall = recall_score(train_label, train_pred, average=None)
    print(train_precision)
    print(train_recall)

    print('\nevaluate the prediction(test data).')
    test_conf = confusion_matrix(test_label, test_pred)
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

    print('\ngenerate report file \t')
    logFile = write_log(paramset, pred_result)
    print(logFile)


if __name__ =="__main__":
    main()
