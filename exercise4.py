from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def main():
    """Main function to run when script is run."""
    data = np.loadtxt("A3_data/bm.csv", delimiter=",")
    np.random.seed(7)
    np.random.shuffle(data)
    X_train, y_train = data[:5000, :-1],  data[:5000, -1]
    X_test, y_test = data[5000:, :-1],  data[5000:, -1]

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(["#ffaaaa", "#aaffaa"]), marker=".")

    n_trees = 100
    n_samples  = 5000
    r  = np.zeros([n_samples, n_trees], dtype=int)
    XX = np.zeros([n_samples, 2, n_trees])
    yy = np.zeros([n_samples, n_trees])
    trees = []

    # XX [rows, cols, tree]
    # yy [rows, tree]
    for i in range(n_trees):
        r[:, i] = np.random.choice(n_samples, size=n_samples, replace=True)
        XX[:, :, i] = X_train[r[:, i], :]
        yy[:, i] = y_train[r[:, i]]
        clf = DecisionTreeClassifier()
        clf.fit(XX[:, :, i], yy[:, i])
        trees.append(clf)
    print("Every tree trained")



if __name__ == "__main__":
    main()
    plt.show()