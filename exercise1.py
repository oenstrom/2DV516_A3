import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit


def main():
    """"""
    data = np.loadtxt("A3_data/mnistsub.csv", delimiter=",")
    train_stop = round(data.shape[0]*0.8)
    X, y = data[:, :-1], data[:, -1]
    X_train, y_train = X[:train_stop, :], y[:train_stop]
    X_vali, y_vali = X[train_stop:, :], y[train_stop:]

    lin = SVC(kernel="linear")
    poly = SVC(kernel="poly")
    rbf = SVC(kernel="rbf")

    lin.fit(X_train, y_train)
    poly.fit(X_train, y_train)
    rbf.fit(X_train, y_train)

    print("Lin Errors:", lin.score(X_vali, y_vali))
    print("Pol Errors:", poly.score(X_vali, y_vali))
    print("RBF Errors:", rbf.score(X_vali, y_vali))

    C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    C_list = [0.001, 0.005, 0.01, 0.05, 0.075, 0.1, 0.5, 1, 5, 10, 50, 100, 150, 250, 500, 1000]
    # C_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 250, 500, 750, 1000]

    best_score = 0
    for C in C_list:
        score = SVC(kernel="linear", C=C).fit(X_train, y_train).score(X_vali, y_vali)
        if score > best_score:
            best_score = score
            best_params = {"C": C}
    print("Linear")
    print(f"  Best score: {best_score}")
    print(f"  Best params: {best_params}")


    best_score = 0
    for C in C_list:
        for gamma in C_list:
            score = SVC(kernel="rbf", C=C, gamma=gamma).fit(X_train, y_train).score(X_vali, y_vali)
            # print(score)
            if score > best_score:
                best_score = score
                best_params = {"C": C, "gamma": gamma}
    print("RBF")
    print(f"  Best score: {best_score}")
    print(f"  Best params: {best_params}")


    degrees = [1, 2, 3, 4, 5, 6]
    best_score = 0
    for d in degrees:
        for C in C_list:
            # for gamma in C_list:
            score = SVC(kernel="poly", degree=d, C=C).fit(X_train, y_train).score(X_vali, y_vali)
            if score > best_score:
                best_score = score
                best_params = {"C": C, "d": d}
    print("Poly")
    print(f"  Best score: {best_score}")
    print(f"  Best params: {best_params}")





    # x_min, x_max = min(X[:, 0]), max(X[:, 0])
    # y_min, y_max = min(X[:, 1]), max(X[:, 1])
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    # grid = np.c_[xx.ravel(), yy.ravel()]
    # z = poly.predict(grid)

    # plt.contour(xx, yy, z.reshape(xx.shape), colors="orange", alpha=0.3)
    # plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], marker=".")
    # plt.scatter(X_train[y_train == 3, 0], X_train[y_train == 3, 1], marker=".")
    # plt.scatter(X_train[y_train == 5, 0], X_train[y_train == 5, 1], marker=".")
    # plt.scatter(X_train[y_train == 9, 0], X_train[y_train == 9, 1], marker=".")

    
    # plt.show()

if __name__ == "__main__":
    main()