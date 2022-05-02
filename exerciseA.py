from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def main():
    """"""
    data = np.loadtxt("A3_data/bm.csv", delimiter=",")
    X, y = data[:, :-1],  data[:, -1]
    n_s = 5000
    np.random.seed(7)
    r = np.random.permutation(len(y))
    X, y = X[r, :], y[r]
    X_s, y_s = X[:n_s, :], y[:n_s]

    svc = SVC(C=20, gamma=0.5)
    svc.fit(X_s, y_s)
    print("Training error:", 1 - svc.score(X_s, y_s) )

    support_vectors = X_s[svc.support_]
    plt.figure("Support vectors + decision boundary and Data + decision boundary", figsize=(12,7))
    plt.subplot(1, 2, 1)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker=".")
    ax = plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = svc.predict(grid)

    # plt.imshow(z.reshape(50, 50), origin="lower", extent=(xlim[0], xlim[1], ylim[0], ylim[1]), cmap=ListedColormap(["#ffaaaa", "#aaffaa"]))
    plt.contour(xx, yy, z.reshape(xx.shape), colors="orange")

    plt.subplot(1, 2, 2)
    plt.scatter(X_s[:, 0], X_s[:, 1], c=y_s, cmap=ListedColormap(["lightblue", "green"]), marker=".")
    plt.contour(xx, yy, z.reshape(xx.shape), colors="orange")
    plt.show()


if __name__ == "__main__":
    main()