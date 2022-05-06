from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main():
    """Main function to run when script is run."""
    data = np.loadtxt("A3_data/bm.csv", delimiter=",")
    np.random.seed(7)
    np.random.shuffle(data)
    X_train, y_train = data[:5000, :-1],  data[:5000, -1]
    X_test, y_test = data[5000:, :-1],  data[5000:, -1]

    margin = 0.5
    grid_size = 500
    x_min, x_max = min(X_train[:, 0]) - margin, max(X_train[:, 0]) + margin
    y_min, y_max = min(X_train[:, 1]) - margin, max(X_train[:, 1]) + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid = np.c_[xx.ravel(), yy.ravel()]

    n_trees = 100
    n_samples  = 5000
    r  = np.zeros([n_samples, n_trees], dtype=int)
    XX = np.zeros([n_samples, 2, n_trees])
    YY = np.zeros([n_samples, n_trees])
    trees = []
    sum_gen_error = 0

    ensemble_boundary = np.zeros(grid_size*grid_size)
    ensemble_pred = np.zeros(y_test.shape[0])
    # XX [rows, cols, tree]
    # yy [rows, tree]
    plt.figure(f"Decision boundary for each of the {n_trees} tree models")
    for i in range(n_trees):
        r[:, i] = np.random.choice(n_samples, size=n_samples, replace=True)
        XX[:, :, i] = X_train[r[:, i], :]
        YY[:, i] = y_train[r[:, i]]
        clf = DecisionTreeClassifier()
        clf.fit(XX[:, :, i], YY[:, i])
        y_pred = clf.predict(X_test)
        ensemble_pred += y_pred
        sum_gen_error += accuracy_score(y_test, y_pred)
        trees.append(clf)
        plt.subplot(10, 10, i + 1)
        p = clf.predict(grid)
        ensemble_boundary += p
        plt.contour(xx, yy, p.reshape(xx.shape), levels=[0.5], colors="orange")
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(left=0.05, bottom=0.01, right=0.95, top=0.95, wspace=0.1, hspace=0.1)

    plt.figure("Ensemble model")
    plt.contour(xx, yy, (ensemble_boundary > (n_trees/2)).reshape(xx.shape), levels=[0.5], colors="orange")


    print("a)")
    print("  Ensemble est. gen. error:", round(accuracy_score(y_test, (ensemble_pred > (n_trees/2))) * 100, 2), "%")
    print("b)")
    print("  Average est. gen. error:", round(sum_gen_error/n_trees * 100, 2), "%")




if __name__ == "__main__":
    main()
    plt.show()