import argparse
from load_dataset import load_data, write_log, algo_map, preprocess_data, show_learncurve
import json
import sys
import importlib
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

def train(args):
    params = vars(args)
    with open(params["config"], mode='r') as f:
        paramset = json.load(f)
    data_dir = paramset["root_dir"]
    samp_rate_t = paramset["sample_rate"]["sample_rate_t"]
    samp_rate_f = paramset["sample_rate"]["sample_rate_f"]

    train_data_raw, train_label_raw, test_data_raw, test_label_raw = load_data(data_dir, samp_rate_t, samp_rate_f)

    if "dimension_reduction" in paramset:
        dim_reducer = paramset["dimension_reduction"]["method"]
        num_components = paramset["dimension_reduction"]["n_components"]
        algo_map[dim_reducer]["parameters"]["n_components"] = num_components
        print('\nbegin dimensionality reduction process.')
        module = importlib.import_module(algo_map[dim_reducer]["module"])
        reducer = getattr(module, algo_map[dim_reducer]["function"])(**algo_map[dim_reducer]["parameters"])
        train_data, train_label, test_data, test_label = preprocess_data(train_data_raw, train_label_raw, test_data_raw, test_label_raw, dim_reducer)
        reducer.fit(train_data)
        train_data = reducer.transform(train_data)
        test_data = reducer.transform(test_data)
        print('\nafter dimensionality reduction:')
        print(train_data.shape)
        print(test_data.shape)

    print('\nbegin training process.')
    classify_method = paramset["classifier"]["method"]
    if "dimension_reduction" not in paramset:
        train_data, train_label, test_data, test_label = preprocess_data(train_data_raw, train_label_raw, test_data_raw, test_label_raw, classify_method)
    module = importlib.import_module(algo_map[classify_method]["module"])
    if algo_map[classify_method]["module"] == "nnet_lib":
        #classifier = getattr(module, algo_map[classify_method]["function"])(train_data, train_label)
        classifier, history = getattr(module, "nnet_training")(train_data, train_label, classify_method, **algo_map[classify_method]["parameters"])
        show_learncurve(history)
    else:
        classifier = getattr(module, algo_map[classify_method]["function"])(**algo_map[classify_method]["parameters"])
        classifier.fit(train_data, train_label)

    print('\npredict for test data.')
    test_pred = classifier.predict(test_data)
    train_pred = classifier.predict(train_data)

    if len(test_pred.shape) > 1:
        test_pred = np.argmax(test_pred, axis=1)
        train_pred = np.argmax(train_pred, axis=1)
        test_label = np.argmax(test_label, axis=1)
        train_label = np.argmax(train_label, axis=1)

    print('\nevaluate the prediction(train data).')
    train_conf = confusion_matrix(train_label, train_pred)
    train_precision = precision_score(train_label, train_pred, average=None)
    train_recall = recall_score(train_label, train_pred, average=None)
    print(train_conf)
    print(train_precision)
    print(train_recall)

    print('\nevaluate the prediction(test data).')
    test_conf = confusion_matrix(test_label, test_pred)
    test_precision = precision_score(test_label, test_pred, average=None)
    test_recall = recall_score(test_label, test_pred, average=None)
    print(test_conf)
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
    #logFile = write_log(paramset, pred_result)
    if algo_map[classify_method]["module"] == "nnet_lib":
        logFile = write_log(paramset, pred_result, classifier, history)
    else:
        logFile = write_log(paramset, pred_result)

    print(logFile)

    if params["show_misclassified"]:
        indices = [i for i in range(len(test_label)) if test_pred[i] != test_label[i]]
        show_data(test_data_raw[indices], test_label[indices], indices, test_pred[indices])


def show_data(data, label, indices, pred=[]):
    fig = plt.figure()
    global cursor
    cursor = 0
    plt.imshow(data[cursor, :, :, 0])
    global category_list
    category_list = ['1 pedestrian', '1 bicyclist', '1 pedestrian and 1 bicyclist', '2 pedestrians', '2 bicyclists']
    title = 'sample index:' + str(indices[cursor]) \
            + ', category: ' + str(category_list[label[cursor]])
    if len(pred) > 0:
        title += ', misclassified as: ' + str(category_list[pred[cursor]])
    plt.title(title)
    fig.canvas.draw()

    def press(event):
        global cursor
        if event.key == 'escape':
            sys.exit(0)
        if event.key == 'left' or event.key == 'up':
            cursor = cursor - 1 if cursor > 0 else 0
        elif event.key == 'right' or event.key == 'down' or event.key == ' ':
            cursor = cursor + 1 if cursor < data.shape[0] - 1 else data.shape[0] - 1
        sys.stdout.flush()
        plt.imshow(data[cursor, :, :, 0])
        global category_list
        title = 'sample index:' + str(indices[cursor]) \
                + ', category: ' + str(category_list[label[cursor]])
        if len(pred) > 0:
            title += ', misclassified as: ' + str(category_list[pred[cursor]])
        plt.title(title)
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', press)
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()


def show(args):
    params = vars(args)
    train_data, train_label, test_data, test_label = load_data(params["datadir"], 1, 1)
    indices = np.arange(params["start"], train_data.shape[0]-1)
    show_data(train_data[params["start"]:], train_label[params["start"]:], indices)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='subcommand help')
    train_parser = subparsers.add_parser('train', help='train help')
    train_parser.add_argument('--config', help='pipeline configuration', type=str)
    train_parser.add_argument('--show-misclassified', help='misclassified samples from test', action='store_true')

    show_parser = subparsers.add_parser('show', help='show help')
    show_parser.add_argument('--datadir', help='data directory', type=str)
    show_parser.add_argument('--start', help='start image index', type=int)

    train_parser.set_defaults(func=train)
    show_parser.set_defaults(func=show)
    args = parser.parse_args()
    args.func(args)

if __name__ =="__main__":
    main()
