import os
from re import I
from venv import create
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import models, layers, utils
import numpy as np
import time
import pandas as pd


def create_models(X_train, y_train):
    """Create two layer and three layer models."""
    # Two layer models
    for units in [50, 100, 200]:
        for units2 in [50, 100, 200]:
            for lr in [0.0001, 0.0005, 0.001]:
                for l2 in [False, True]:
                    model = models.Sequential()
                    model.add(layers.InputLayer(input_shape=(28, 28, 1)))
                    model.add(layers.Flatten())
                    if l2:
                        model.add(layers.Dense(units, activation="relu", kernel_regularizer="l2"))
                        model.add(layers.Dense(units2, activation="relu", kernel_regularizer="l2"))
                    else:
                        model.add(layers.Dense(units, activation="relu"))
                        model.add(layers.Dense(units2, activation="relu"))
                    model.add(layers.Dense(10, activation="softmax"))
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
                    model.fit(X_train, y_train, epochs=10)
                    model.save(f"saved_models/two_layers-{units}x{units2}_lr-{lr}_l2-{l2}")
    
    # Three layer models
    for units in [50, 100, 200]:
        for units2 in [50, 100, 200]:
            for units3 in [50, 100, 200]:
                for lr in [0.0001, 0.0005, 0.001]:
                    for l2 in [False, True]:
                        model = models.Sequential()
                        model.add(layers.InputLayer(input_shape=(28, 28, 1)))
                        model.add(layers.Flatten())
                        if l2:
                            model.add(layers.Dense(units, activation="relu", kernel_regularizer="l2"))
                            model.add(layers.Dense(units2, activation="relu", kernel_regularizer="l2"))
                            model.add(layers.Dense(units3, activation="relu", kernel_regularizer="l2"))
                        else:
                            model.add(layers.Dense(units, activation="relu"))
                            model.add(layers.Dense(units2, activation="relu"))
                            model.add(layers.Dense(units3, activation="relu"))
                        model.add(layers.Dense(10, activation="softmax"))
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
                        model.fit(X_train, y_train, epochs=10)
                        model.save(f"saved_models/three_layers-{units}x{units2}x{units3}_lr-{lr}_l2-{l2}")


def read_saved_models():
    # max_score = [0]
    saved_models = []
    for x in next(os.walk("models"))[1]:
        saved_models.append(tf.keras.models.load_model(f"models/{x}"))
    return saved_models

def plot_16(X_train, y_train):
    for i, img_nr in enumerate(np.random.choice(X_train.shape[0], size=16, replace=False)):
        plt.subplot(4, 4, i+1)
        plt.gca().set_title(f"Label: {y_train[img_nr].argmax()}")
        plt.imshow(X_train[img_nr], cmap="gray")
        plt.tight_layout()
    plt.show()

def main():
    """Main function to run when the script is run."""
    data = np.array(pd.read_csv("A3_data/fashion-mnist_train.csv"))
    X_train, y_train = (data[:, 1:]/255.0).reshape(-1, 28, 28), tf.keras.utils.to_categorical(data[:, 0])
    X_val, y_val = X_train[48000:], y_train[48000:]
    X_train, y_train = X_train[:48000], y_train[:48000]
    data = np.array(pd.read_csv("A3_data/fashion-mnist_test.csv"))
    X_test, y_test = (data[:, 1:]/255.0).reshape(-1, 28, 28), tf.keras.utils.to_categorical(data[:, 0])

    # plot_16(X_train, y_train)

    # Just run this if you really want to stare at models being trained...
    # create_models(X_train, y_train)

    saved_models = read_saved_models()


    max_score = [0]
    for model in saved_models:
        _, acc = model.evaluate(X_val, y_val, verbose=0)
        if acc > max_score[0]:
            max_score = [acc, model]
        print(f"{model}: {acc*100} %")

    print("------")
    print(max_score)
    _, acc = max_score[1].evaluate(X_test, y_test)
    max_score[1].summary()
    print(f"Score: {acc*100} %")
    exit()



    # for lr in [0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.009]:
    #     model = models.Sequential()
    #     model.add(layers.InputLayer(input_shape=(28, 28, 1)))
    #     model.add(layers.Flatten())
    #     model.add(layers.Dense(200, activation="relu"))
    #     model.add(layers.Dense(100, activation="relu"))
    #     model.add(layers.Dense(10, activation="softmax"))
    #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    #     model.fit(X_train, y_train, epochs=10)
    #     print("-------------")
    #     _, acc = model.evaluate(X_test, y_test, verbose=0)
    #     print(f"{lr}, Score:", acc*100, "%")
    #     print("-------------")

    # for u1 in [50, 100, 300]:
    #     for u2 in [50, 100, 300]:
    #         for u3 in [50, 100, 300]:
    #             model = models.Sequential()
    #             model.add(layers.InputLayer(input_shape=(28, 28, 1)))
    #             model.add(layers.Flatten())
    #             model.add(layers.Dense(u1, activation="relu"))
    #             model.add(layers.Dense(u2, activation="relu"))
    #             model.add(layers.Dense(u3, activation="relu"))
    #             model.add(layers.Dense(10, activation="softmax"))
    #             model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007), loss="categorical_crossentropy", metrics=["accuracy"])
    #             model.fit(X_train, y_train, epochs=10)
    #             print("-------------")
    #             _, acc = model.evaluate(X_test, y_test, verbose=0)
    #             print(f"{u1}, {u2}, {u3} | Score: {acc*100} %")
    #             print("-------------")

    s = 0
    for i in range(10):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(28, 28, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(300, activation="relu"))
        model.add(layers.Dense(100, activation="relu"))
        model.add(layers.Dense(100, activation="relu"))
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=10)
        # print("-------------")
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        s += acc
    print("Avg. score:", (s/len(range(10)))*100 )

    exit()

    model = models.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.Flatten(),
        # layers.Dense(784, activation="relu"),
        # layers.Dense(130, activation="relu"),
        # layers.Dense(50, activation="relu"),
        # layers.Flatten(),
        # layers.Dense(790, activation="relu"),
        # layers.Dense(130, activation="relu"),
        # layers.Dense(50, activation="relu"),
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


if __name__ == "__main__":
    main()