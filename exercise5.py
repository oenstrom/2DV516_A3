import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import models, layers, utils
import numpy as np
import time
import pandas as pd

data = np.array(pd.read_csv("A3_data/fashion-mnist_train.csv"))
X_train, y_train = (data[:, 1:]/255.0).reshape(-1, 28, 28), tf.keras.utils.to_categorical(data[:, 0])
data = np.array(pd.read_csv("A3_data/fashion-mnist_test.csv"))
X_test, y_test = (data[:, 1:]/255.0).reshape(-1, 28, 28), tf.keras.utils.to_categorical(data[:, 0])


# for i, img_nr in enumerate(np.random.choice(X_train.shape[0], size=16, replace=False)):
#     plt.subplot(4, 4, i+1)
#     plt.gca().set_title(f"Label: {y_train[img_nr].argmax()}")
#     plt.imshow(X_train[img_nr], cmap="gray")
# plt.tight_layout()
# plt.show()

model = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Flatten(),
    # layers.Dense(784, activation="relu"),
    # layers.Dense(130, activation="relu"),
    # layers.Dense(50, activation="relu"),
    layers.Dense(790, activation="relu"),
    layers.Dense(130, activation="relu"),
    layers.Dense(50, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0009), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10)

_, acc = model.evaluate(X_test, y_test)

print("----------------------------------------------")
print(f"Accuracy: {acc * 100.0} %")
exit()




# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", input_shape=(28, 28, 1)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(150, activation="relu", kernel_initializer="he_uniform"),
#     layers.Dense(10, activation="softmax")
# ])

model = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Dense(175, activation="relu", kernel_initializer="he_uniform"),
    layers.Flatten(),
    layers.Dense(120, activation="relu", kernel_initializer="he_uniform"),
    layers.Dense(10, activation="softmax")
])


opt = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10)
_, acc = model.evaluate(X_test, y_test)

print("----------------------------------------------")
print(f"Accuracy: {acc * 100.0} %")