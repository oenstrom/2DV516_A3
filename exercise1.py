import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def main():
    """"""
    data = np.loadtxt("A3_data/mnistsub.csv", delimiter=",")
    train_stop = round(data.shape[0]*0.8)
    X, y = data[:, :-1], data[:, -1]
    X_train, y_train = X[:train_stop, :], y[:train_stop]
    X_vali, y_vali = X[train_stop:, :], y[train_stop:]

    kernels = {
        "linear": {"C": [0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 10, 50, 100, 200]},
        "rbf": {"C": [0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 10, 50, 100, 200], "gamma": [0.001, 0.01, 0.05, 0.075, 0.1, 0.5, 0.75, 1, 5, 10, 100]},
        "poly": {"degree": [2, 3, 4, 5, 6], "C": [0.01, 0.1, 1, 10], "gamma": [0.01, 0.05, 0.075, 0.1]}
    }

    # Grid search
    for kernel, params in kernels.items():
        best_score = 0
        for degree in params.get("degree", [1]):
            for C in params.get("C", [1]):
                for gamma in params.get("gamma", [1]):
                    clf = SVC(kernel=kernel, degree=degree, C=C, gamma=gamma)
                    score = clf.fit(X_train, y_train).score(X_vali, y_vali)
                    if score > best_score:
                        best_score = score
                        params["best"] = clf

    margin = 1
    grid_size = 500
    x_min, x_max = min(X[:, 0]) - margin, max(X[:, 0]) + margin
    y_min, y_max = min(X[:, 1]) - margin, max(X[:, 1]) + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid = np.c_[xx.ravel(), yy.ravel()]

    lin = kernels["linear"]["best"]
    rbf = kernels["rbf"]["best"]
    pol = kernels["poly"]["best"]

    plt.figure("Best models with decision boundary and data", figsize=(12, 7))
    plt.subplot(1, 3, 1)
    plt.gca().set_title(f"Linear\nC={lin.C}\nScore: {round(lin.score(X_vali, y_vali), 3)}")
    plt.imshow(lin.predict(grid).reshape(xx.shape), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap="Pastel2")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker=".", cmap="Dark2")

    plt.subplot(1, 3, 2)
    plt.gca().set_title(f"RBF\nC={rbf.C}, gamma={rbf.gamma}\nScore: {round(rbf.score(X_vali, y_vali), 3)}")
    plt.imshow(rbf.predict(grid).reshape(xx.shape), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap="Pastel2")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker=".", cmap="Dark2")

    plt.subplot(1, 3, 3)
    plt.gca().set_title(f"Poly\nC={pol.C}, gamma={pol.gamma}, d={pol.degree}\nScore: {round(pol.score(X_vali, y_vali), 3)}")
    plt.imshow(pol.predict(grid).reshape(xx.shape), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap="Pastel2")
    # plt.contour(xx, yy, pol.predict(grid).reshape(xx.shape))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker=".", cmap="Dark2")
    
    plt.show()

if __name__ == "__main__":
    main()