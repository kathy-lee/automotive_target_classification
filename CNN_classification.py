import numpy as np
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.initializers import RandomUniform, GlorotUniform, he_uniform
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard, ModelCheckpoint
from load_dataset import load_data, preprocess_data, plot_learncurve
from sklearn.model_selection import train_test_split
from time import process_time
from lr_finder import LRFinder


data_dir = '/home/kangle/dataset/PedBicCarData'
train_data, train_label, test_data, test_label = load_data(data_dir, 2, 2, 6, 1)
train_data, train_label, test_data, test_label = preprocess_data(train_data, train_label, test_data, test_label, 'cnn_a')

train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1, random_state=42)
print("Split training data into training and validation data:\n")
print("training data: %d" % train_data.shape[0])
print("validation data: %d" % val_data.shape[0])

train_data -= np.mean(train_data, axis=0)
#train_data /= np.std(train_data, axis=0)
val_data -= np.mean(val_data, axis=0)
#val_data /= np.std(val_data, axis=0)
test_data -= np.mean(test_data, axis=0)
#test_data /= np.std(test_data, axis=0)
print("After standardization:\n")

model = models.Sequential()
regularizer = None#l2(1e-4)
initializer = GlorotUniform()#RandomUniform()#
#model.add(BatchNormalization())
model.add(Conv2D(32, [3, 3], input_shape=train_data.shape[1:], activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1'))
model.add(MaxPooling2D())
model.add(Conv2D(64, [3, 3],  activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2'))
model.add(MaxPooling2D())
model.add(Conv2D(128, [3, 3],  activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3'))
model.add(MaxPooling2D())
model.add(Conv2D(256, [3, 3],  activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(240, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, name='dense_1'))
model.add(Dropout(0.5))
model.add(Dense(168, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer, name='dense_2'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax', name='dense_3'))

model.summary()

def piecewise_constant_fn(epoch):
    if epoch < 10:
        return 0.0005/2
    elif epoch < 20:
        return 0.0002/2
    else:
        return 0.0001/2

lr_scheduler = LearningRateScheduler(piecewise_constant_fn)
earlystop_callback = EarlyStopping(monitor='val_loss', patience=10)
log_dir = "/home/kangle/Projects/radar_object_classification"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
opt = Adam()
# opt = SGD(learning_rate=0.1, momentum=0.9, decay=1e-2/epochs)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

lr_finder = LRFinder(model, stop_factor=4)
steps = np.ceil(len(train_label)/32)
lr_finder.find((train_data, train_label), steps_per_epoch=steps, start_lr=1e-6, lr_mult=1.01, batch_size=32)
lr_finder.plot_loss()

# history = model.fit(train_data,
#                     train_label,
#                     epochs=30,
#                     batch_size=16,
#                     verbose=2,
#                     validation_data=(val_data, val_label),
#                     callbacks=[lr_scheduler, tensorboard_callback])
#
# # load the saved best model
# # model = models.load_model('best_model.h5')
#
# # evaluate model
# test_pred = model.predict(test_data)
#
# t_start = process_time()
# _,acc = model.evaluate(test_data, test_label, batch_size=16, verbose=2)
# t_end = process_time()
# t_cost = t_end - t_start
# print(f"Test Accuracy: {acc:.4f}, Inference time: {t_cost:.2f}s")
#
#
# plot_learncurve("CNN", history=history)
