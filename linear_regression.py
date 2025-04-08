import numpy as np
from gradient import finite_difference_gradient

def basic_linear_regression(X, y, X_val, y_val, bias, steps):
    if (bias == True):
        X = np.c_[np.ones(X.shape[0]), X]
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y # OLS
    if bias == True:
        X_val = np.c_[np.ones(X_val.shape[0]), X_val]
    y_pred = X_val @ theta
    loss = np.mean((y_pred - y_val) ** 2) * 0.5
    return theta, np.full(steps, loss)

def GD_linear_regression_finite_diff(X, y, X_val, y_val, alpha, k, epsilon):
    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros((X.shape[1], 1))
    for i in range(k):
        grad = finite_difference_gradient(X, y, theta, epsilon)
        theta -= alpha * grad
    return theta, []

def GD_linear_regression(X, y, X_val, y_val, alpha, k, steps):
    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros((X.shape[1], 1))
    X_val = np.c_[np.ones(X_val.shape[0]), X_val]
    loss = np.zeros(steps)

    for i in range(1, k + 1):
        y_pred = X @ theta
        er = y_pred - y
        gd = np.mean(X * er, axis=0).reshape((-1, 1))
        theta -= alpha * gd
        if i % (k // steps) == 0:
            y_pred_val = X_val @ theta
            loss[i // (k // steps) - 1] = np.mean((y_pred_val - y_val) ** 2) * 0.5
    return theta, loss

def SGD_linear_regression(X, y, X_val, y_val, alpha, k, batch_size, steps):
    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros((X.shape[1], 1))
    X_val = np.c_[np.ones(X_val.shape[0]), X_val]
    loss = np.zeros(steps)
    num_batches = max(X.shape[0] // batch_size, 1)

    for i in range(1, k + 1):
        perm = np.random.permutation(X.shape[0])
        X_perm = X[perm]
        y_perm = y[perm]
        for (X_batch, y_batch) in zip(
                np.array_split(X_perm, num_batches),
                np.array_split(y_perm, num_batches)):
            y_pred = X_batch @ theta
            er = y_pred - y_batch
            gd = np.mean(X_batch * er, axis=0).reshape((-1, 1))
            theta -= alpha * gd
            if i % (k // steps) == 0:
                y_pred_val = X_val @ theta
                loss[i // (k // steps) - 1] = np.mean((y_pred_val - y_val) ** 2) * 0.5
    return theta, loss