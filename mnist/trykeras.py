import numpy as np
import pandas as pd
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, \
    accuracy_score, confusion_matrix
from tensorflow.keras import models, layers, callbacks, optimizers
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

sub = 10_000
X_train = X_train[:sub,:,:]
X_test = X_test[:sub,:,:]
y_train = y_train[:sub]
y_test = y_test[:sub]

# normalize!!!
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

n, w, h = X_train.shape

print(n, w, h)

num_classes = 10

model = models.Sequential()
layer1 = 512
layer2 = 512
# layer3 = 50
batch_size = 2500
dropout = 0.2
model.add(layers.Dense(layer1, input_dim=w*h, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(dropout))

model.add(layers.Dense(layer2, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(dropout))

# model.add(layers.Dense(layer3, activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(dropout))
#
model.add(layers.Dense(num_classes, activation='softmax'))

learning_rate = 0.15
# opt = optimizers.Adam(lr=learning_rate)
opt = optimizers.RMSprop()

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

# callback = callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train.reshape(n,w*h), y_train,
                    epochs=150,
                    validation_data=(X_test.reshape(n,w*h), y_test),
                    batch_size=batch_size,
                    verbose=1
                    )

y_pred = model.predict(X_test.reshape(n,w*h))
y_pred = np.argmax(y_pred, axis=1)
val_accur = accuracy_score(y_test, y_pred)
print("Keras validation accuracy", val_accur)

conf = confusion_matrix(y_test, y_pred)
print(conf)

if True: # Show training result
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")

    accur = history.history['accuracy']
    plt.plot(accur, label='train_accuracy')
    val_accur = history.history['val_accuracy']
    plt.plot(val_accur, label='valid_accuracy')
    plt.title(f"batch_size {batch_size}, Layers {layer1,layer2}\ntrain {accur[-1]:.3f}, valid {val_accur[-1]:.3f}, learning_rate={learning_rate:.2f}\ndropout {dropout:.2f}")
    plt.xlim(0, 300)
    plt.ylim(0.5, 1.02)
    plt.legend(loc='lower right')
    plt.show()
