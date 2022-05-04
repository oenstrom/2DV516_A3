from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def main():
    """Main function to run when script is run."""
    data = np.loadtxt("A3_data/bm.csv", delimiter=",")
    # np.random.seed(7)
    # np.random.shuffle(data)
    X_train, y_train = data[:5000, :-1],  data[:5000, -1]
    X_test, y_test = data[5000:, :-1],  data[5000:, -1]

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(["#ffaaaa", "#aaffaa"]), marker=".")

    # plt.show()

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print(clf.predict([[1.95, 0.45]]))

    # n_trees = 100
    # n  = 5000
    n_trees = 2
    n  = 4
    r  = np.zeros([n, n_trees], dtype=int)
    XX = np.zeros([n, 2, n_trees])
    yy = np.zeros([n, n_trees])
    trees = []

    # XX [träd, :, set]
    # yy [träd, set]
    X_train = X_train[:4]
    for i in range(n_trees):
        r[:, i] = np.random.choice(n, size=n, replace=True)
        XX[:, :, i] = X_train[r[:, i], :]
        yy[:, i] = y_train[r[:, i]]
        clf = DecisionTreeClassifier()
        clf.fit(XX[i, :, :], yy[i, :])
        trees.append(clf)



if __name__ == "__main__":
    main()