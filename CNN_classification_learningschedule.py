import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization, ReLU, AveragePooling2D, Softmax
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.optimizers import schedules, Adam
import numpy as np
from load_dataset import load_data, preprocess_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from time import process_time

data_dir = '/home/kangle/dataset/PedBicCarData'
train_data, train_label, test_data, test_label = load_data(data_dir, 1, 1, 6, 1)

train_data, train_label, test_data, test_label = preprocess_data(train_data, train_label, test_data, test_label, 'cnn_a')

train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1, random_state=42)
print("Split training data into training and validation data:\n")
print(train_data.shape)
print(val_data.shape)

model = models.Sequential()
model.add(Conv2D(16, [10, 10], input_shape=train_data.shape[1:], kernel_initializer='glorot_uniform', padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(10,10), strides=2))

model.add(Conv2D(32, [5, 5], kernel_initializer='glorot_uniform', padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(10,10), strides=2))

model.add(Conv2D(32, [5, 5], kernel_initializer='glorot_uniform', padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(10,10), strides=2))

model.add(Conv2D(32, [5, 5], kernel_initializer='glorot_uniform', padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(5,5), strides=2))

model.add(Conv2D(32, [5, 5], kernel_initializer='glorot_uniform', padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

model.add(Flatten())
model.add(Dense(5, activation='softmax'))

step = tf.Variable(0, trainable=False)
boundaries = [1562, 3125]
values = [0.01, 0.001, 0.0001]
learning_rate_fn = schedules.PiecewiseConstantDecay(boundaries, values)
learning_rate_customized = learning_rate_fn(step)
optimizer_customized = Adam(lr=learning_rate_customized)

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_data, train_label,
                    epochs=30,
                    batch_size=128,
                    verbose=2,
                    validation_data=(val_data, val_label),
                    shuffle=True)

# evaluate model
t_start = process_time()
_,acc = model.evaluate(test_data, test_label, batch_size=128, verbose=2)
t_end = process_time()
t_cost = t_end - t_start
print(f"Test Accuracy: {acc:.4f}, Inference time: {t_cost:.2f}s")
