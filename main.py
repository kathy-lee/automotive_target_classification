import argparse
from load_dataset import load_data, write_log, algo_map
import json
import numpy as np
import sys
import importlib
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#global cursor

def train(args):
    params = vars(args)
    with open(params["config"], mode='r') as f:
        paramset = json.load(f)
    data_dir = paramset["root_dir"]
    samp_rate_t = paramset["sample_rate"]["sample_rate_t"]
    samp_rate_f = paramset["sample_rate"]["sample_rate_f"]

    train_data_raw, train_label, test_data_raw, test_label = load_data(data_dir, samp_rate_t, samp_rate_f)

    train_data = np.reshape(train_data_raw, (train_data_raw.shape[0], -1))
    print(train_data.shape)
    test_data = np.reshape(test_data_raw, (test_data_raw.shape[0], -1))
    print(test_data.shape)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    if "dimension_reduction" in paramset:
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

    if params["show_misclassified"]:
        indices = [i for i in range(len(test_label)) if test_pred[i] != test_label[i]]
        fig = plt.figure()
        global cursor
        cursor = 0
        plt.imshow(test_data_raw[indices[cursor], :, :, 0])
        category_list = ['1 pedestrian', '1 bicyclist', '1 pedestrian and 1 bicyclist', '2 pedestrians', '2 bicyclists']
        title = 'sample index:' + str(indices[cursor]) \
                    + ', true category: ' + str(category_list[test_label[indices[cursor]] - 1]) \
                    + ', misclassified as:' + str(category_list[test_pred[indices[cursor]] - 1])
        plt.title(title)
        fig.canvas.draw()

        def press(event):
            global cursor
            if event.key == 'escape':
                sys.exit(0)
            if event.key == 'left' or event.key == 'up':
                cursor = cursor - 1 if cursor > 0 else 0
            elif event.key == 'right' or event.key == 'down' or event.key == ' ':
                cursor = cursor + 1 if cursor < len(indices) - 1 else len(indices) - 1
            sys.stdout.flush()
            plt.imshow(test_data_raw[indices[cursor], :, :, 0])
            category_list = ['1 pedestrian', '1 bicyclist', '1 pedestrian and 1 bicyclist', '2 pedestrians',
                             '2 bicyclists']
            title = 'true category: ' + str(category_list[test_label[indices[cursor]] - 1]) \
                    + ', misclassified as:' + str(category_list[test_pred[indices[cursor]] - 1])
            plt.title(title)
            # plt.show()
            fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', press)
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show()


def show(args):
    params = vars(args)
    train_data, train_label, test_data, test_label = load_data(params["datadir"], 1, 1)
    fig = plt.figure()
    #global cursor
    cursor = 0
    plt.imshow(train_data[params["start"], :, :, 0])
    category_list = ['1 pedestrian', '1 bicyclist', '1 pedestrian and 1 bicyclist', '2 pedestrians', '2 bicyclists']
    plt.title(category_list[train_label[params["start"]] - 1])
    #plt.show()
    fig.canvas.draw()

    def press(event):
        global cursor
        if event.key == 'escape':
            sys.exit(0)
        if event.key == 'left' or event.key == 'up':
            cursor = cursor - 1 if cursor > 0 else 0
        elif event.key == 'right' or event.key == 'down' or event.key == ' ':
            cursor = cursor + 1 if cursor < train_data.shape[0] - 1 else train_data.shape[0] - 1
        sys.stdout.flush()
        plt.imshow(train_data[cursor, :, :, 0])
        category_list = ['1 pedestrian', '1 bicyclist', '1 pedestrian and 1 bicyclist', '2 pedestrians', '2 bicyclists']
        plt.title(category_list[train_label[cursor] - 1])
        #plt.show()
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', press)
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()


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
