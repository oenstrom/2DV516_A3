import os
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from keras import models, layers
import numpy as np
import pandas as pd


def create_models(X_train, y_train):
    """Create two layer and three layer models. Not the most beautiful code...."""
    # Two layer models
    for units in [50, 100, 200]:
        for units2 in [50, 100, 200]:
            for lr in [0.0001, 0.0005, 0.001]:
                for drop in [False, True]:
                    model = models.Sequential()
                    model.add(layers.InputLayer(input_shape=(28, 28, 1)))
                    model.add(layers.Flatten())
                    if drop:
                        model.add(layers.Dense(units, activation="relu"))
                        model.add(layers.Dropout(0.4))
                        model.add(layers.Dense(units2, activation="relu"))
                    else:
                        model.add(layers.Dense(units, activation="relu"))
                        model.add(layers.Dense(units2, activation="relu"))
                    model.add(layers.Dense(10, activation="softmax"))
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
                    model.fit(X_train, y_train, epochs=10)
                    model.save(f"saved_models/two_layers-{units}x{units2}_lr-{lr}_drop-{drop}")
    
    # Three layer models
    for units in [50, 100, 200]:
        for units2 in [50, 100, 200]:
            for units3 in [50, 100, 200]:
                for lr in [0.0001, 0.0005, 0.001]:
                    for drop in [False, True]:
                        model = models.Sequential()
                        model.add(layers.InputLayer(input_shape=(28, 28, 1)))
                        model.add(layers.Flatten())
                        if drop:
                            model.add(layers.Dense(units, activation="relu", kernel_regularizer="l2"))
                            model.add(layers.Dropout(0.4))
                            model.add(layers.Dense(units2, activation="relu", kernel_regularizer="l2"))
                            model.add(layers.Dropout(0.2))
                            model.add(layers.Dense(units3, activation="relu", kernel_regularizer="l2"))
                        else:
                            model.add(layers.Dense(units, activation="relu"))
                            model.add(layers.Dense(units2, activation="relu"))
                            model.add(layers.Dense(units3, activation="relu"))
                        model.add(layers.Dense(10, activation="softmax"))
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
                        model.fit(X_train, y_train, epochs=10)
                        model.save(f"saved_models/three_layers-{units}x{units2}x{units3}_lr-{lr}_drop-{drop}")


def read_saved_models(path="saved_models"):
    """Read already trained models."""
    saved_models = []
    for x in next(os.walk(path))[1]:
        saved_models.append(tf.keras.models.load_model(f"{path}/{x}"))
    return saved_models

def plot_16(X_train, y_train, labels, size=16):
    """Plot random samples from the training set."""
    plt.figure(f"{size} random samples from training set")
    for i, img_nr in enumerate(np.random.choice(X_train.shape[0], size=size, replace=False)):
        plt.subplot(4, 4, i+1)
        plt.gca().set_title(f"{labels[y_train[img_nr].argmax()]} ({y_train[img_nr].argmax()})")
        plt.imshow(X_train[img_nr], cmap=plt.cm.binary)
    plt.tight_layout()

def main():
    """Main function to run when the script is run."""
    data = np.array(pd.read_csv("A3_data/fashion-mnist_train.csv"))
    # np.random.seed(7)
    np.random.shuffle(data)
    X_train, y_train = (data[:, 1:]/255.0).reshape(-1, 28, 28), tf.keras.utils.to_categorical(data[:, 0])
    data = np.array(pd.read_csv("A3_data/fashion-mnist_test.csv"))
    X_test, y_test = (data[:, 1:]/255.0).reshape(-1, 28, 28), tf.keras.utils.to_categorical(data[:, 0])
    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    plot_16(X_train, y_train, labels, 16)

    # Just run this if you really want to stare at models being trained...
    # create_models(X_train, y_train)

    # Read saved models and evaluate them
    print("Reading saved models...")
    saved_models = read_saved_models()
    max_score = [0]
    for model in saved_models:
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        if acc > max_score[0]:
            max_score = [acc, model]
        print(f"{model}: {acc*100} %")

    print("------------------------------------------------------------------------------------------")
    print("----------------------------------------Best model----------------------------------------")
    max_score[1].summary()
    y_pred1 = max_score[1].predict(X_test).argmax(1)
    print(classification_report(y_test.argmax(1), y_pred1, target_names=labels, digits=4))
    print("------------------------------------------------------------------------------------------")


    # https://blog.tensorflow.org/2018/04/fashion-mnist-with-tfkeras.html
    model = tf.keras.Sequential([
        layers.Conv2D(64, 2, padding="same", activation="relu", input_shape=(28,28,1)),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        layers.Conv2D(32, 2, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])


    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10)
    y_pred2 = model.predict(X_test).argmax(1)
    print(classification_report(y_test.argmax(1), y_pred2, target_names=labels, digits=4))


    plt.figure("Confusion matrices")
    ax1 = plt.subplot(1, 2, 1)
    ax1.title.set_text("Non-CNN FNN")
    ax2 = plt.subplot(1, 2, 2)
    ax2.title.set_text("CNN")
    ConfusionMatrixDisplay.from_predictions(y_test.argmax(1), y_pred1, labels=np.unique(y_test.argmax(1)), display_labels=labels, ax=ax1, xticks_rotation="vertical", colorbar=False)
    ConfusionMatrixDisplay.from_predictions(y_test.argmax(1), y_pred2, labels=np.unique(y_test.argmax(1)), display_labels=labels, ax=ax2, xticks_rotation="vertical", colorbar=False)
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.99, top=0.99, wspace=0.18, hspace=0.1)


if __name__ == "__main__":
    main()
    plt.show()