import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from mnist_load import mnist
from threading import Thread
import time

def one_vs_all(clf, X_train, y_train, X_test):
    """One-vs-All classification"""
    labels = np.unique(y_train)
    prob_table = np.zeros((len(X_test), len(labels)))

    def _calc_prob(k):
        """Fit the data for all the different binary classifiers and save decision values in table."""
        clf.fit(X_train, y_train==k)
        prob_table[:, k] = np.ravel(clf.decision_function(X_test))

    threads = []
    for k in labels:
        t = Thread(target=_calc_prob, args=(k,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()

    return labels[prob_table.argmax(1)]

def main():
    """Main function to run when script is run."""
    train_images, train_labels, X_test, y_test = mnist("A3_data")

    train_size = 10000
    X_train = train_images[:train_size]
    y_train = train_labels[:train_size]
    val_size = 1000
    X_val = train_images[train_size:train_size + val_size]
    y_val = train_labels[train_size:train_size + val_size]
    

    params = {
        "C": np.linspace(1, 10, 10),
        "gamma": np.linspace(0.001, 0.1, 10)
    }
    best_rbf = GridSearchCV(estimator=SVC(kernel="rbf"), param_grid=params, scoring="accuracy", n_jobs=10, cv=3)
    best_rbf.fit(X_val, y_val)
    best_C, best_gamma = best_rbf.best_params_["C"], best_rbf.best_params_["gamma"]
    print(f"Best params: C={best_C}, gamma={best_gamma}")

    rbf = SVC(kernel="rbf", C=best_C, gamma=best_gamma, random_state=42)
    rbf.fit(X_train, y_train)
    print("OvO Train accuracy:", rbf.score(X_train, y_train) * 100, "%")
    print("OvO Test accuracy:", rbf.score(X_test, y_test) * 100, "%")

    start_time = time.time()
    clf = SVC(kernel="rbf", C=best_C, gamma=best_gamma, random_state=42)
    y_pred = one_vs_all(clf, X_train, y_train, X_test)

    print("OvR Test accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
    print("Time:", time.time() - start_time, "seconds")

    labels = np.unique(y_train)
    plt.figure("Confusion matrices")
    ax1 = plt.subplot(1, 2, 1)
    ax1.title.set_text("Built in One-vs-One")
    ax2 = plt.subplot(1, 2, 2)
    ax2.title.set_text("Custom One-vs-All")
    ConfusionMatrixDisplay.from_estimator(rbf, X_test, y_test, labels=labels, ax=ax1)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=labels, ax=ax2)

    # from sklearn.multiclass import OneVsOneClassifier
    # start_time = time.time()
    # clf = OneVsOneClassifier(SVC(kernel="rbf", C=best_C, gamma=best_gamma, random_state=42))
    # clf.fit(X_train, y_train)
    # Y_pred = clf.predict(X_test)
    # print("Built-in OvO Test accuracy:", accuracy_score(y_test, Y_pred) * 100, "%")
    # print("Time:", time.time() - start_time, "seconds")

    plt.show()




if __name__ == "__main__":
    main()