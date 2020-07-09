import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization, LSTM
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1,l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from time import process_time
import matplotlib.pyplot as plt

def nnet_training(train_data, train_label, nnet, optimizer, learning_rate, loss, metrics, batch_size, epochs):
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1,
                                                                    random_state=42)
    print( nnet )
    model = globals()[nnet](train_data)
    opt = getattr(tf.keras.optimizers, optimizer)(learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=[metrics])
    history = model.fit(train_data,
                        train_label,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(val_data, val_label))
    return model, history

def cnn_a(train_data):
    # train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1, random_state=42)
    # print("Split training data into training and validation data:\n")
    # print("training data: %d" % train_data.shape[0])
    # print("validation data: %d" % val_data.shape[0])

    model = models.Sequential()
    model.add(Conv2D(16, [5, 5], input_shape=train_data.shape[1:], activation='relu', kernel_initializer='he_uniform',
              kernel_regularizer='l2', name='conv_1'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, [5, 5], activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2', name='conv_2'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu', kernel_initializer='he_uniform', name='dense_1'))
    # model.add(BatchNormalization())
    model.add(Dense(84, activation='relu', kernel_initializer='he_uniform', name='dense_2'))
    # model.add(BatchNormalization())
    model.add(Dense(5, activation='softmax', name='dense_3'))

    # opt = Adam(learning_rate=0.01)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # history = model.fit(train_data,
    #                     train_label,
    #                     epochs=5,
    #                     batch_size=128,
    #                     verbose=2,
    #                     validation_data=(val_data, val_label))
    return model

def alexnet():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')]) #  for imagenet:1000
    return model

def rnn_a(train_data):
    # train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1,
    #                                                                 random_state=42)
    # print("Split training data into training and validation data:\n")
    # print("training data: %d" % train_data.shape[0])
    # print("validation data: %d" % val_data.shape[0])

    model = models.Sequential()
    model.add(LSTM(64, input_shape=(train_data.shape[1], train_data.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))

    # opt = Adam(learning_rate=0.01)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # history = model.fit(train_data,
    #                     train_label,
    #                     epochs=10,
    #                     batch_size=128,
    #                     verbose=2,
    #                     validation_data=(val_data, val_label))
    return model

