import argparse
import json
import sys
import importlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from load_dataset import load_data, write_log, algo_map, preprocess_data, plot_learncurve, load_model, nnet_fit


def train(args):
    params = vars(args)
    with open(params["config"], mode='r') as f:
        paramset = json.load(f)
    data_dir = paramset["root_dir"]

    if "sample_rate" in paramset:
        samp_rate_t = paramset["sample_rate"]["sample_rate_t"]
        samp_rate_f = paramset["sample_rate"]["sample_rate_f"]
    else:
        samp_rate_t = 1
        samp_rate_f = 1
    data_bunch = load_data(data_dir, samp_rate_t, samp_rate_f)
    data_bunch_visual = np.copy(data_bunch)

    if "dimension_reduction" in paramset:
        dim_reducer = paramset["dimension_reduction"]["method"]
        num_components = paramset["dimension_reduction"]["n_components"]
        algo_map[dim_reducer]["parameters"]["n_components"] = num_components
        print('\nbegin dimensionality reduction process.')
        module = importlib.import_module(algo_map[dim_reducer]["module"])
        reducer = getattr(module, algo_map[dim_reducer]["function"])(**algo_map[dim_reducer]["parameters"])
        data_bunch = preprocess_data(data_bunch, dim_reducer)
        reducer.fit(data_bunch["train_data"])
        data_bunch["train_data"] = reducer.transform(data_bunch["train_data"])
        data_bunch["test_data"] = reducer.transform(data_bunch["test_data"])
        print('\nafter dimensionality reduction:')
        print(data_bunch["train_data"].shape)
        print(data_bunch["test_data"].shape)

    classify_method = paramset["classifier"]["method"]
    classify_parameter = paramset["classifier"]["parameter"]
    if classify_method == "cnn" or "rnn":
        data_bunch = preprocess_data(data_bunch, classify_method)
        classifier = load_model(paramset["classifier"]["model"], data_bunch["train_data"].shape)
        history = nnet_fit(data_bunch, classifier, paramset["classifier"]["parameter"])
        plot_learncurve(classify_method, history)
    else:
        module = importlib.import_module(algo_map[classify_method]["module"])
        classifier = getattr(module, algo_map[classify_method]["function"])(**classify_parameter)
        classifier.fit(data_bunch["train_data"], data_bunch["train_label"])
        plot_learncurve(classify_method, estimator=classifier, data=data_bunch["train_data"],
                        label=data_bunch["train_label"], train_sizes=np.linspace(0.05, 0.2, 5))

    print('\npredict for test data.')
    test_pred = classifier.predict(data_bunch["test_data"])
    train_pred = classifier.predict(data_bunch["train_data"])
    train_label = data_bunch["train_label"]
    test_label = data_bunch["test_label"]
    if len(test_pred.shape) > 1:
        test_pred = np.argmax(test_pred, axis=1)
        train_pred = np.argmax(train_pred, axis=1)
        test_label = np.argmax(data_bunch["test_label"], axis=1)
        train_label = np.argmax(data_bunch["train_label"], axis=1)

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
    if classify_method == "cnn" or "rnn":
        log_file = write_log(paramset, pred_result, classifier, history)
    else:
        log_file = write_log(paramset, pred_result)
    print(log_file)

    if params["show_misclassified"]:
        indices = [i for i in range(len(data_bunch["test_label"])) if test_pred[i] != data_bunch["test_label"][i]]
        show_data(data_bunch_visual["test_data"][indices], data_bunch["test_label"][indices], indices, test_pred[indices])


def show_data(data, label, indices, pred=None):
    fig = plt.figure()
    global CURSOR
    global CATEGORY_LIST

    CURSOR = 0
    plt.imshow(data[CURSOR, :, :, 0])
    CATEGORY_LIST = ['1 pedestrian', '1 bicyclist', '1 pedestrian and 1 bicyclist', '2 pedestrians', '2 bicyclists']
    title = 'sample index:' + str(indices[CURSOR]) + ', category: ' + str(CATEGORY_LIST[label[CURSOR]])
    if pred is not None:
        title += ', misclassified as: ' + str(CATEGORY_LIST[pred[CURSOR]])
    plt.title(title)
    fig.canvas.draw()

    def press(event):
        global CURSOR
        global CATEGORY_LIST
        if event.key == 'escape':
            sys.exit(0)
        if event.key == 'left' or event.key == 'up':
            CURSOR = CURSOR - 1 if CURSOR > 0 else 0
        elif event.key == 'right' or event.key == 'down' or event.key == ' ':
            CURSOR = CURSOR + 1 if CURSOR < data.shape[0] - 1 else data.shape[0] - 1
        sys.stdout.flush()
        plt.imshow(data[CURSOR, :, :, 0])
        title = 'sample index:' + str(indices[CURSOR]) + ', category: ' + str(CATEGORY_LIST[label[CURSOR]])
        if pred is not None:
            title += ', misclassified as: ' + str(CATEGORY_LIST[pred[CURSOR]])
        plt.title(title)
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', press)
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()


def show(args):
    params = vars(args)
    data_bunch = load_data(params["datadir"], 1, 1)
    if params["type"] == "train":
        indices = np.arange(params["start"], data_bunch["train_data"].shape[0]-1)
        show_data(data_bunch["train_data"][params["start"]:], data_bunch["train_label"][params["start"]:], indices)
    else:
        indices = np.arange(params["start"], data_bunch["test_data"].shape[0]-1)
        show_data(data_bunch["test_data"][params["start"]:], data_bunch["test_label"][params["start"]:], indices)

        
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='subcommand help')
    train_parser = subparsers.add_parser('train', help='train help')
    train_parser.add_argument('--config', help='pipeline configuration', type=str)
    train_parser.add_argument('--show-misclassified', help='misclassified samples from test', action='store_true')

    show_parser = subparsers.add_parser('show', help='show help')
    show_parser.add_argument('--datadir', help='data directory', type=str)
    show_parser.add_argument('--start', help='start image index', type=int)
    show_parser.add_argument('--type', help='training data or test data', type=str)

    train_parser.set_defaults(func=train)
    show_parser.set_defaults(func=show)
    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        parser.print_help()
        parser.exit()


if __name__ == "__main__":
    main()
