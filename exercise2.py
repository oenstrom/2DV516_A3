import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from mnist_load import mnist
from threading import Thread

def calc_prob(prob_table, clf, X_train, y_train, X_test, k):
    print(f"Running {k} in thread")
    clf.fit(X_train, y_train==k)
    prob_table[:, k] = clf.predict_proba(X_test)[:, 1]


def main():
    """"""
    train_images, train_labels, X_test, y_test = mnist("A3_data")

    train_size = 20000
    X_train = train_images[:train_size]
    y_train = train_labels[:train_size]
    val_size = 1000
    X_val = train_images[train_size:train_size + val_size]
    y_val = train_labels[train_size:train_size + val_size]
    
    # params = {
    #     "C": [0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 10, 50, 100, 200],
    #     "gamma": [0.001, 0.01, 0.05, 0.075, 0.1, 0.5, 0.75, 1, 5, 10, 100]
    # }
    params = {
        # "C": [0.00001, 0.0001, 0.001, 0.005, 1, 5, 10, 50, 100, 150, 200],
        # "C": [0.01, 0.1, 1, 10, 100],
        "C": np.linspace(1, 10, 10),
        # "C": np.linspace(40, 50, 50),
        # "gamma": [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.75, 1, 2, 3, 4, 5]
        # "gamma": np.linspace(0.00000001, 5, 50)
        # "gamma": [0.01, 0.1, 1, 10, 100]
        "gamma": np.linspace(0.001, 0.1, 10)
    }
    # rbf = SVC(kernel="rbf")
    # print(rbf.C, rbf.gamma)
    print("loaded data")
    # best_rbf = GridSearchCV(estimator=SVC(kernel="rbf"), param_grid=params, scoring="accuracy", n_jobs=8, cv=3)
    # best_rbf.fit(X_val, y_val)
    # print()
    # print("Train score:", best_rbf.best_score_)
    # print(best_rbf.best_params_)
    # print("Test score:", best_rbf.score(X_test, y_test))


    # rbf = SVC(kernel="rbf", C=10, gamma=0.01)
    # rbf = SVC(kernel="rbf", C=3, gamma=0.023)
    # rbf.fit(X_train, y_train)
    # print("Train score:", rbf.score(X_train, y_train))
    # print("Test score:", rbf.score(X_test, y_test))

    my_classes = np.unique(y_test)
    # X_test_short = X_test[2:5]
    prob_table = np.zeros((len(X_test), len(my_classes)))

    threads = []
    for k in my_classes:
        clf = SVC(kernel="rbf", C=3, gamma=0.023, probability=True, class_weight="balanced", random_state=42)
        t = Thread(target=calc_prob, args=(prob_table, clf, X_train, y_train, X_test, k))
        t.start()
        threads.append(t)
        # clf.fit(X_train, y_train==k)
        # prob_table[:, k] = clf.predict_proba(X_test_short)[:, 1]
    
    for t in threads:
        t.join()

    print(prob_table)

    y_pred = my_classes[prob_table.argmax(1)]
    print(y_pred)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    exit()


    my_classes = np.array([0, 1])
    X_test_short = X_test[2:5]
    prob_table = np.zeros((len(X_test_short), len(my_classes)))
    
    clf_0 = SVC(kernel="rbf", C=3, gamma=0.023, probability=True, class_weight="balanced", random_state=42)
    clf_0.fit(X_train, y_train==0)
    prob_table[:, 0] = clf_0.predict_proba(X_test_short)[:, 1]

    clf_1 = SVC(kernel="rbf", C=3, gamma=0.023, probability=True, class_weight="balanced", random_state=42)
    clf_1.fit(X_train, y_train==1)
    prob_table[:, 1] = clf_1.predict_proba(X_test_short)[:, 1]

    print(prob_table)

    y_pred = my_classes[prob_table.argmax(1)]
    print(y_pred)

    # print(np.unique(y_train))
    # for k in range(0, 10):
    #     SVC(kernel="rbf", C=3, gamma=0.023, probability=True, random_state=42)

    clf_2 = SVC(kernel="rbf", C=3, gamma=0.023)
    clf_3 = SVC(kernel="rbf", C=3, gamma=0.023)
    clf_4 = SVC(kernel="rbf", C=3, gamma=0.023)
    clf_5 = SVC(kernel="rbf", C=3, gamma=0.023)
    clf_6 = SVC(kernel="rbf", C=3, gamma=0.023)
    clf_7 = SVC(kernel="rbf", C=3, gamma=0.023)
    clf_8 = SVC(kernel="rbf", C=3, gamma=0.023)
    clf_9 = SVC(kernel="rbf", C=3, gamma=0.023)





if __name__ == "__main__":
    main()