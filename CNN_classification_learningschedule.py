import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization, ReLU, AveragePooling2D, Softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import schedules, Adam
import numpy as np
from load_dataset import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from time import process_time

data_dir = '/home/kangle/dataset/PedBicCarData'
train_data, train_label, test_data, test_label = load_data(data_dir, 1, 2)
train_label = to_categorical(train_label-1, num_classes=5)
test_label = to_categorical(test_label-1, num_classes=5)

print("Data sample distribution in training set: %d %d %d %d %d\n" % (np.count_nonzero(train_label==1),
      np.count_nonzero(train_label==2), np.count_nonzero(train_label==3),
      np.count_nonzero(train_label==4), np.count_nonzero(train_label==5)))
print("Data sample distribution in test set: %d %d %d %d %d\n" % (np.count_nonzero(test_label==1),
      np.count_nonzero(test_label==2), np.count_nonzero(test_label==3),
      np.count_nonzero(test_label==4), np.count_nonzero(test_label==5)))

train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1, random_state=42)
print("Split training data into training and validation data:\n")
print("training data: %d" % train_data.shape[0])
print("validation data: %d" % val_data.shape[0])

model = models.Sequential()
model.add(Conv2D(16, [10, 10], input_shape=train_data.shape[1:], kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(10,10), strides=2))

model.add(Conv2D(32, [5, 5], padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(10,10), strides=2))

model.add(Conv2D(32, [5, 5], padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(10,10), strides=2))

model.add(Conv2D(32, [5, 5], padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(10,10), strides=2))

model.add(Conv2D(32, [5, 5], padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

model.add(Dense(5, activation='softmax', name='dense_3'))
model.add(Softmax())

step = tf.Variable(0, trainable=False)
boundaries = [1280, 2560]
values = [0.01, 0.001, 0.0001]
learning_rate_fn = schedules.PiecewiseConstantDecay(boundaries, values)
learning_rate_customized = learning_rate_fn(step)
optimizer_customized = Adam(lr=learning_rate_customized)

model.compile(optimizer=optimizer_customized,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

results = model.fit(train_data,
                    train_label,
                    epochs=30,
                    batch_size=128,
                    verbose=2,
                    validation_data=(val_data, val_label))

# evaluate model
t_start = process_time()
_,acc = model.evaluate(test_data, test_label, batch_size=128, verbose=2)
t_end = process_time()
t_cost = t_end - t_start
print(f"Test Accuracy: {acc:.4f}, Inference time: {t_cost:.2f}s")
