import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from mnist_load import mnist
from threading import Thread
import time

def calc_prob(prob_table, clf, X_train, y_train, X_test, k):
    clf.fit(X_train, y_train==k)
    # prob_table[:, k] = clf.predict_proba(X_test)[:, 1]
    prob_table[:, k] = np.ravel(clf.decision_function(X_test))


def main():
    """"""
    train_images, train_labels, X_test, y_test = mnist("A3_data")

    train_size = 50000
    X_train = train_images[:train_size]
    y_train = train_labels[:train_size]
    val_size = 1000
    X_val = train_images[train_size:train_size + val_size]
    y_val = train_labels[train_size:train_size + val_size]
    

    params = {
        "C": np.linspace(1, 10, 10),
        "gamma": np.linspace(0.001, 0.1, 10)
    }
    # rbf = SVC(kernel="rbf")
    # print(rbf.C, rbf.gamma)
    print("loaded data")
    best_rbf = GridSearchCV(estimator=SVC(kernel="rbf"), param_grid=params, scoring="accuracy", n_jobs=10, cv=3)
    best_rbf.fit(X_val, y_val)
    print(best_rbf.best_params_)

    # rbf = SVC(kernel="rbf", C=best_rbf.best_params_["C"], gamma=best_rbf.best_params_["gamma"], random_state=42)
    # rbf.fit(X_train, y_train)
    # print("OvO Train accuracy:", rbf.score(X_train, y_train) * 100, "%")
    # print("OvO Test accuracy:", rbf.score(X_test, y_test) * 100, "%")

    mnist_classes = np.unique(y_test)
    prob_table = np.zeros((len(X_test), len(mnist_classes)))

    threads = []
    start_time = time.time()
    for k in mnist_classes:
        # clf = SVC(kernel="rbf", C=best_rbf.best_params_["C"], gamma=best_rbf.best_params_["gamma"], probability=True, random_state=42)
        clf = SVC(kernel="rbf", C=best_rbf.best_params_["C"], gamma=best_rbf.best_params_["gamma"], random_state=42)
        clf.fit(X_train, y_train==k)
        # print(np.ravel(clf.decision_function(X_test[:4])))
        # print(clf.predict_proba(X_test[:4])[:, 1])
        # exit()
        t = Thread(target=calc_prob, args=(prob_table, clf, X_train, y_train, X_test, k))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()

    # print(prob_table)

    y_pred = mnist_classes[prob_table.argmax(1)]
    # print(y_pred)
    print("OvR Test accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
    print("Time:", time.time() - start_time, "seconds")

    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.multiclass import OneVsOneClassifier
    start_time = time.time()
    clf = OneVsRestClassifier(SVC(kernel="rbf", C=best_rbf.best_params_["C"], gamma=best_rbf.best_params_["gamma"], random_state=42))
    clf.fit(X_train, y_train)
    Y_pred = clf.predict(X_test)
    print("Built-in OvR Test accuracy:", accuracy_score(y_test, Y_pred) * 100, "%")
    print("Time:", time.time() - start_time, "seconds")

    # clf = OneVsOneClassifier(SVC(kernel="rbf", C=best_rbf.best_params_["C"], gamma=best_rbf.best_params_["gamma"], random_state=42))
    # clf.fit(X_train, y_train)
    # Y_pred = clf.predict(X_test)
    # print("Built-in OvO Test accuracy:", accuracy_score(y_test, Y_pred) * 100, "%")





if __name__ == "__main__":
    main()