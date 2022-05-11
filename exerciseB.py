from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd


data = np.array(pd.read_csv("A3_data/fashion-mnist_train.csv"))
X_train, y_train = data[:, 1:], data[:, 0]
X_train = X_train / 255.0

data = np.array(pd.read_csv("A3_data/fashion-mnist_test.csv"))
X_test, y_test = data[:, 1:], data[:, 0]
X_test = X_test / 255.0


clf = MLPClassifier(learning_rate_init=0.01, verbose=True, hidden_layer_sizes=[100, 100, 100])
clf.fit(X_train, y_train)


score = clf.score(X_test, y_test)
print("------------------------------------")
print(f"Score: {score * 100} %")


# import tensorflow as tf
# from keras import layers, models, datasets, losses


# print("Tensorflow version:", tf.__version__)

# mnist = datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = models.Sequential([
#     layers.Flatten(input_shape=(28, 28)),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(10)
# ])

# predictions = model(x_train[:1]).numpy()
# loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

# model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)

# model.evaluate(x_test,  y_test, verbose=2)