from os.path import join as pjoin
from sklearn.model_selection import learning_curve, train_test_split
from lr_finder import LRFinder
import keras as K
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import importlib


algo_map = {
    'pca': {"module": "sklearn.decomposition", "function": "PCA",
            "parameters": {"svd_solver": "randomized", "whiten": True}},
    'lda': {"module": "sklearn.discriminant_analysis", "function": "LinearDiscriminantAnalysis"},
    'ica': {"module": "sklearn.decomposition", "function": "FastICA"},
    'lr':  {"module": "sklearn.linear_model", "function": "LogisticRegression",
            "parameters": {}},
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
                "parameters": {}},
    'multi_step': {"module": "multistep_lr", "function": "MultiStepLR"},
    'one_cycle': {"module": "cyclic_lr", "function": "CyclicLR"},
    'cyclic': {"modeule": "cyclic_lr", "fuction": ""},
    'sgdr': {"module": "sgdr", "function": "SGDRScheduler"}
}

nn_type = ['cnn', 'cnn']

def read_file(index, data_type, root_dir):
    file_index = "{0:02d}".format(index)
    if index == 0:
        file_name = data_type + 'NoCar' + '.h5'
    else:
        file_name = data_type + 'NoCar_' + file_index + '.h5'
    file_name = pjoin(root_dir, file_name)
    print(file_name)
    f = h5py.File(file_name, 'r')
    key = list(f.keys())[0]
    data = np.array(f[key])
    return data


def load_data(root_dir, samp_rate_t=1, samp_rate_f=1, file_num_train=20, file_num_test=5):
    print('load the data.')
    train_data = np.array([])

    for i in range(1,file_num_train+1):
        data_block = read_file(i, 'trainData', root_dir)
        data_block = data_block[:,:,0::samp_rate_t,0::samp_rate_f]
        train_data = np.vstack([train_data, data_block]) if train_data.size else data_block

    test_data = np.array([])
    for i in range(1,file_num_test+1):
        data_block = read_file(i, 'testData', root_dir)
        data_block = data_block[:, :, 0::samp_rate_t, 0::samp_rate_f]
        test_data = np.vstack([test_data, data_block]) if test_data.size else data_block

    train_data = np.transpose(train_data, (0, 3, 2, 1))
    test_data = np.transpose(test_data, (0, 3, 2, 1))

    # read label
    train_label = read_file(0, 'trainLabel', root_dir)
    train_label = train_label.flatten()
    train_label = train_label[:file_num_train*1000]
    train_label -= 1
    test_label = read_file(0, 'testLabel', root_dir)
    test_label = test_label.flatten()
    test_label = test_label[:file_num_test*1000]
    test_label -= 1

    print(train_data.shape)
    print(test_data.shape)

    print("\nData sample distribution in training set: %d %d %d %d %d" % (np.count_nonzero(train_label == 1),
                                                                          np.count_nonzero(train_label == 2),
                                                                          np.count_nonzero(train_label == 3),
                                                                          np.count_nonzero(train_label == 4),
                                                                          np.count_nonzero(train_label == 0)))
    print("Data sample distribution in test set: %d %d %d %d %d" % (np.count_nonzero(test_label == 1),
                                                                    np.count_nonzero(test_label == 2),
                                                                    np.count_nonzero(test_label == 3),
                                                                    np.count_nonzero(test_label == 4),
                                                                    np.count_nonzero(test_label == 0)))
    dataset = {
        "train_data": train_data,
        "train_label": train_label,
        "test_data": test_data,
        "test_label": test_label
    }
    return dataset


def preprocess_data(dataset, classify_method):

    if classify_method.lower() in [
        'pca', 'lda', 'ica', 'lr', 'svm', 'decision tree', 'knn',
        'random forest', 'ada boost', 'gradient boost', 'xgboost'
    ]:
        print('\nReformat data for ML classifier:')
        dataset["train_data"] = dataset["train_data"].reshape(dataset["train_data"].shape[0], -1)
        dataset["test_data"] = dataset["test_data"].reshape(dataset["test_data"].shape[0], -1)
    elif classify_method.lower() in ['cnn', 'rnn']:
        print('\nReformat data for neural network classifier:')
        train_label = K.utils.to_categorical(dataset["train_label"], num_classes=5)
        test_label = K.utils.to_categorical(dataset["test_label"], num_classes=5)
        train_data = dataset["train_data"]
        test_data = dataset["test_data"]
        if classify_method.lower() in ['rnn']:
            train_data = np.squeeze(train_data)
            test_data = np.squeeze(test_data)
            train_data = np.transpose(train_data, (0, 2, 1))
            test_data = np.transpose(test_data, (0, 2, 1))
        print(train_data.shape)
        print(test_data.shape)
        train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.2,
                                                                        random_state=42)
        print("\nSplit training data into training and validation data:")
        print("training data: %d" % train_data.shape[0])
        print("validation data: %d" % val_data.shape[0])
        dataset = {
            "train_data": train_data,
            "train_label": train_label,
            "val_data": val_data,
            "val_label": val_label,
            "test_data": test_data,
            "test_label": test_label
        }
        dataset = normalize(dataset)
    # scaler = StandardScaler()
    # train_data = scaler.fit_transform(train_data)
    # test_data = scaler.fit_transform(test_data)
    return dataset


def normalize(dataset):
    train_stats_mean = dataset["train_data"].mean()
    train_stats_std = dataset["train_data"].std()
    dataset["train_data"] -= train_stats_mean
    dataset["train_data"] /= train_stats_std
    dataset["val_data"] -= train_stats_mean
    dataset["val_data"] /= train_stats_std
    dataset["test_data"] -= train_stats_mean
    dataset["test_data"] /= train_stats_std
    print("\nAfter normalization:")
    print("training data: mean %f, std %f" % (dataset["train_data"].mean(), dataset["train_data"].std()))
    print("validation data: mean %f, std %f" % (dataset["val_data"].mean(), dataset["val_data"].std()))
    print("test data: mean %f, std %f" % (dataset["test_data"].mean(), dataset["test_data"].std()))
    return dataset


def write_log(paramset, result, classifier=None, history=None):
    epoch_time = int(time.time())
    filename = "log" + str(epoch_time) + ".txt"
    with open(filename, "w+") as f:
        # write configuration parameters
        current_time = time.strftime("%d. %m. %Y. %H:%M:%S\n\n", time.localtime())
        f.write(current_time)
        f.write("resample in time axis: %d\n" % paramset["sample_rate"]["sample_rate_t"])
        f.write("resample in frequency axis: %d\n\n" % paramset["sample_rate"]["sample_rate_f"])
        if "dimension_reduction" in paramset:
            f.write("dimension reduction method: %s\n" % paramset["dimension_reduction"]["method"])
            f.write("dimension after reduction: %d\n" % paramset["dimension_reduction"]["n_components"])
            f.write("%s parameters: \n" % paramset["dimension_reduction"]["method"])
            for key, value in algo_map[paramset["dimension_reduction"]["method"]]["parameters"].items():
                f.write("\t%s : %s\n" % (key,str(value)))
        f.write("classifier method: %s\n" % paramset["classifier"]["method"])
        f.write("%s parameters: \n" % paramset["classifier"]["method"])
        # for key, value in algo_map[paramset["classifier"]["method"]]["parameters"].items():
        #     f.write("\t%s : %s\n" % (key,str(value)))
        for key, value in paramset["classifier"]["parameter"].items():
            f.write("\t%s : %s\n" % (key,str(value)))

        # write train/test results
        f.write("\nTRAINING PERFORMANCE: \n")
        f.write("confusion matrix: \n")
        train_conf = np.array_str(result["train_conf"])
        f.write(train_conf)
        train_precision = np.array_str(result["train_precision"])
        f.write("\naverage precision score: %s\n" % train_precision)
        train_recall = np.array_str(result["train_recall"])
        f.write("average recall score: %s\n" % train_recall)

        f.write("\nTEST PERFORMANCE: \n")
        f.write("confusion matrix: \n")
        test_conf = np.array_str(result["test_conf"])
        f.write(test_conf)
        test_precision = np.array_str(result["test_precision"])
        f.write("\naverage precision score: %s\n" % test_precision)
        test_recall = np.array_str(result["test_recall"])
        f.write("average recall score: %s\n" % test_recall)

        if classifier:
            f.write("\ntraining accuracy:\n")
            f.write(' '.join(map(str, history.history['accuracy'])))
            f.write("\ntraining loss:\n")
            f.write(' '.join(map(str, history.history['loss'])))
            f.write("\nvalidation accuracy:\n")
            f.write(' '.join(map(str, history.history['val_accuracy'])))
            f.write("\nvalidation loss: \n" )
            f.write(' '.join(map(str, history.history['val_loss'])))
            f.write("\n\n")
            #classifier.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("MODEL ARCHITECTURE:\n")
            for layer in paramset["classifier"]["model"]:
                for key, value in layer.items():
                    f.write("%s : %s\t" % (key, str(value)))
                f.write("\n")
    return filename


def plot_learncurve(title, history=None, estimator=None, data=None, label=None, train_sizes=None):
    if estimator is None:
        plt.figure()
        plt.subplot(211)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train', 'test'], loc='lower right')
        plt.grid(True)
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc='upper right')
        plt.grid(True)
        plt.suptitle('Model Accuracy and Loss with %s' % title)
        plt.tight_layout()
        plt.show()
    else:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, data, label, cv=5, n_jobs=1, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.figure()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid("on")
        plt.title(title)
        plt.show()


def without_keys(dic, keys):
    return {x: dic[x] for x in dic if x not in keys}


def load_model(model_layers, data_shape):
    print('\nload nn model.')
    nn_input = K.layers.Input(data_shape[1:])
    for i, layer_dict in enumerate(model_layers):
        para = without_keys(layer_dict, {"name", "type"})
        layer = getattr(K.layers, layer_dict["type"])(**para)
        #layer_class_ = getattr(K.layers, layer_dict["type"])(**para)
        #layer = layer_class_(para)
        if i == 0:
            x = layer(nn_input)
        else:
            x = layer(x)

    model = K.models.Model(inputs=nn_input, outputs=x)
    return model


def nnet_fit(dataset, model, train_para):

    if "learning_rate" in train_para:
        opt = getattr(K.optimizers, train_para["optimizer"])(train_para["learning_rate"])
    else:
        opt = getattr(K.optimizers, train_para["optimizer"])
    model.compile(optimizer=opt, loss=train_para["loss"], metrics=[train_para["metrics"]])

    steps = np.ceil(len(dataset["train_label"]) / train_para["batch_size"])

    # learning rate range test
    # lr_finder = LRFinder(model, stop_factor=4)
    # lr_finder.find((dataset["train_data"], dataset["train_label"]), steps_per_epoch=steps, start_lr=1e-6,
    #                lr_mult=1.01, batch_size=train_para["batch_size"])
    # lr_finder.plot_loss()

    if "learning_rate_policy" in train_para:
        lr_policy = train_para["learning_rate_policy"]
        module = importlib.import_module(algo_map[lr_policy]["module"])
        lr_scheduler = getattr(module, algo_map[lr_policy]["function"])(**train_para["learning_rate_schedule"])
    else:
        lr_scheduler = None
    print('\nBegin training process.')
    history = model.fit(dataset["train_data"],
                        dataset["train_label"],
                        epochs=train_para["epochs"],
                        batch_size=train_para["batch_size"],
                        verbose=2,
                        validation_data=(dataset["val_data"], dataset["val_label"]),
                        callbacks=[lr_scheduler])
    if 'lr' in lr_scheduler.history:
        plt.figure()
        plt.plot(lr_scheduler.history['lr'])
        plt.xlabel('iterations')
        plt.ylabel('learning rate')
        plt.title('Learning Rate Schedule')
        plt.show(block=False)

    return history


def piecewise_constant_decay(epoch, base_lr, step_size, decay_rate):
    lr = base_lr / pow(decay_rate, epoch//step_size)
    return lr