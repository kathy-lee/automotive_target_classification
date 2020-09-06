import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, Model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization, AveragePooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.initializers import RandomUniform, GlorotUniform, he_uniform
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint, Callback
from tensorflow.keras.backend import get_value, set_value
from train_helper import load_data, preprocess_data, plot_learncurve
from sklearn.model_selection import train_test_split
from time import process_time
from lr_finder import LRFinder
from sgdr import SGDRScheduler
from cyclic_lr import CyclicLR


data_dir = '/home/kangle/dataset/PedBicCarData'
data_bunch = load_data(data_dir, 2, 2, 6, 1)
data_bunch = preprocess_data(data_bunch, 'cnn')

model = models.Sequential()
regularizer = None#l2(1e-4)
initializer = he_uniform()
model.add(Conv2D(32, [3, 3], input_shape=data_bunch["train_data"].shape[1:], activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name='conv_1'))
model.add(MaxPooling2D())
model.add(Conv2D(64, [3, 3],  activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name='conv_2'))
model.add(MaxPooling2D())
model.add(Conv2D(128, [3, 3],  activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name='conv_3'))
model.add(MaxPooling2D())
model.add(Conv2D(256, [3, 3],  activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name='conv_4'))
model.add(MaxPooling2D())
model.add(Conv2D(256, [3, 3],  activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name='conv_5'))
model.add(AveragePooling2D())
model.add(Flatten())
#model.add(GlobalAveragePooling2D())
#model.add(Dense(512, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, name='dense_1'))
#model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, name='dense_2'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax', name='dense_3'))

model.summary()

opt = Adam(learning_rate=1e-3)
#opt = SGD(learning_rate=0.04, momentum=0.9, decay=1e-2)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

num_batchsize = 128
num_epochs = 30
steps = np.ceil(len(data_bunch["train_label"])/num_batchsize)

# learning rate range test
# lr_finder = LRFinder(model, stop_factor=4)
# lr_finder.find((train_data, train_label), steps_per_epoch=steps, start_lr=1e-6, lr_mult=1.01, batch_size=num_batchsize)
# lr_finder.plot_loss()

# SGDR learning rate policy
# set learning rate range according to lr range test result
# min_lr = 5e-5
# max_lr = 2e-3
# lr_scheduler = SGDRScheduler(min_lr, max_lr, steps, lr_decay=1.0, cycle_length=1, mult_factor=2)

# one cycle learning rate policy
# set max learning rate according to lr range test result
max_lr = 3e-4
#lr_scheduler = CyclicLR(base_lr=max_lr/10, max_lr=max_lr, step_size=np.ceil(steps*num_epochs/2), max_momentum=0.95, min_momentum=0.85)

def piecewise_constant_fn(epoch):
    if epoch < 10:
        return 3e-4
    elif epoch < 20:
        return 1e-4
    else:
        return 5e-5

lr_scheduler = LearningRateScheduler(piecewise_constant_fn)

class LearningRate_History(Callback):
    def __init__(self):
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        self.history.setdefault('lr', []).append(get_value(self.model.optimizer.lr))

lr_history = LearningRate_History()

#lr_scheduler = ReduceLROnPlateau()

earlystop_callback = EarlyStopping(monitor='val_loss', patience=10)
log_dir = "/home/kangle/Projects/radar_object_classification"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

history = model.fit(data_bunch["train_data"],
                    data_bunch["train_label"],
                    epochs=num_epochs,
                    batch_size=num_batchsize,
                    verbose=2,
                    validation_data=(data_bunch["val_data"], data_bunch["val_label"]),
                    callbacks=[lr_scheduler, lr_history])
#models.save_model(model, 'best_model000')

# load the saved best model
# model = models.load_model('best_model.h5')

# evaluate model
test_pred = model.predict(data_bunch["test_data"])

t_start = process_time()
_,acc = model.evaluate(data_bunch["test_data"], data_bunch["test_label"], batch_size=num_batchsize, verbose=2)
t_end = process_time()
t_cost = t_end - t_start
print(f"Test Accuracy: {acc:.4f}, Inference time: {t_cost:.2f}s")

plot_learncurve("CNN", history=history)

if 'lr' in lr_history.history:
    plt.plot(lr_history.history['lr'])
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.title('Learning Rate Schedule')
    plt.show()
else:
    raise ValueError("no lr info in history.")

if 'lr' in lr_scheduler.history:
    plt.plot(lr_scheduler.history['lr'])
    plt.xlabel('iterations')
    plt.ylabel('learning rate')
    plt.title('Learning Rate Schedule')
    plt.show()
else:
    raise ValueError("no lr info in history.")