import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, Model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization, AveragePooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.initializers import RandomUniform, GlorotUniform, he_uniform
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.applications import resnet50
from load_dataset import load_data, preprocess_data, plot_learncurve
from sklearn.model_selection import train_test_split
from time import process_time
from lr_finder import LRFinder
from sgdr import SGDRScheduler
from cyclic_lr import CyclicLR


data_dir = '/home/kangle/dataset/PedBicCarData'
train_data, train_label, test_data, test_label = load_data(data_dir, 2, 2, 6, 1)
train_data, train_label, test_data, test_label = preprocess_data(train_data, train_label, test_data, test_label, 'cnn_a')

train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.2, random_state=42)
print("\nSplit training data into training and validation data:")
print("training data: %d" % train_data.shape[0])
print("validation data: %d" % val_data.shape[0])

train_stats_mean = train_data.mean()
train_stats_std = train_data.std()
train_data -= train_stats_mean
train_data /= train_stats_std
val_data -= train_stats_mean
val_data /= train_stats_std
test_data -= train_stats_mean
test_data /= train_stats_std
print("\nAfter normalization:")
print("training data: mean %f, std %f" % (train_data.mean(), train_data.std()))
print("validation data: mean %f, std %f" % (val_data.mean(), val_data.std()))
print("test data: mean %f, std %f" % (test_data.mean(), test_data.std()))

train_data = np.repeat(train_data, 3, axis=3)
val_data = np.repeat(val_data, 3, axis=3)
test_data = np.repeat(test_data, 3, axis=3)
print(train_data.shape)

base_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=train_data.shape[1:])
base_model.trainable = False
inputs = Input(shape=train_data.shape[1:])
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(5)(x)
model = Model(inputs, outputs)

model.summary()

opt = Adam(learning_rate=1e-3)
#opt = SGD(learning_rate=0.04, momentum=0.9, decay=1e-2)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

num_batchsize = 128
num_epochs = 30
steps = np.ceil(len(train_label)/num_batchsize)

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
max_lr = 1e-3
lr_scheduler = CyclicLR(base_lr=max_lr/10, max_lr=max_lr, step_size=np.ceil(steps*num_epochs/2), max_momentum=0.95, min_momentum=0.85)

def piecewise_constant_fn(epoch):
    if epoch < 10:
        return 1e-3
    elif epoch < 20:
        return 5e-4
    else:
        return 3e-4

#lr_scheduler = LearningRateScheduler(piecewise_constant_fn)

#lr_scheduler = ReduceLROnPlateau()

earlystop_callback = EarlyStopping(monitor='val_loss', patience=10)
log_dir = "/home/kangle/Projects/radar_object_classification"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

history = model.fit(train_data,
                    train_label,
                    epochs=num_epochs,
                    batch_size=num_batchsize,
                    verbose=2,
                    validation_data=(val_data, val_label))
#models.save_model(model, 'best_model000')

# unfreeze and fine tuning the model
# base_model.trainable = True
# opt = Adam(learning_rate=1e-5)
# #opt = SGD(learning_rate=0.04, momentum=0.9, decay=1e-2)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_data,
#           train_label,
#           epochs=num_epochs,
#           batch_size=num_batchsize,
#           verbose=2,
#           validation_data=(val_data, val_label),
#           callbacks=[])

# load the saved best model
# model = models.load_model('best_model.h5')

# evaluate model
test_pred = model.predict(test_data)

t_start = process_time()
_,acc = model.evaluate(test_data, test_label, batch_size=num_batchsize, verbose=2)
t_end = process_time()
t_cost = t_end - t_start
print(f"Test Accuracy: {acc:.4f}, Inference time: {t_cost:.2f}s")

plot_learncurve("CNN", history=history)

if 'lr' in lr_scheduler.history:
    plt.plot(lr_scheduler.history['lr'])
    plt.xlabel('iterations')
    plt.ylabel('learning rate')
    plt.title('Learning Rate Schedule')
    plt.show()
else:
    raise ValueError("no lr info in history.")