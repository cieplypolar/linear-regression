import numpy as np
from numpy.ma.core import shape

from gradient import finite_difference_gradient
from loss import ols

def basic_analytic_linear_regression(X, y, X_val, y_val, intercept):
    if intercept:
        X = np.c_[np.ones(X.shape[0]), X]

    theta = np.linalg.pinv(X.T @ X) @ X.T @ y # OLS

    if intercept:
        X_val = np.c_[np.ones(X_val.shape[0]), X_val]

    y_pred = X_val @ theta
    val_loss = np.mean((y_pred - y_val) ** 2) * 0.5
    train_loss = np.mean((X @ theta - y) ** 2) * 0.5
    return theta, val_loss, train_loss

def GD_OLS_finite_difference_linear_regression(X, y, X_val, y_val, alpha, epochs, epsilon):
    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros((X.shape[1], 1))
    for i in range(epochs):
        grad = finite_difference_gradient(X, y, theta, epsilon)
        theta -= alpha * grad
    return theta, [], []

#
def GD_linear_regression(X, y, X_val, y_val, grad, alpha, epochs, steps, loss=ols):
    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros((X.shape[1], 1))

    X_val = np.c_[np.ones(X_val.shape[0]), X_val]

    val_loss = np.zeros(steps)
    train_loss = np.zeros(steps)

    for i in range(1, epochs + 1):
        gd = grad(X, y, theta)
        theta -= alpha * gd
        if i % (epochs // steps) == 0:
            val_loss[i // (epochs // steps) - 1] = loss(X_val, theta, y_val)
            train_loss[i // (epochs // steps) - 1] = loss(X, theta, y)
    return theta, val_loss, train_loss

def SGD_linear_regression(X, y, X_val, y_val, grad, alpha, epochs, batch_size, steps, loss=ols):
    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros((X.shape[1], 1))
    X_val = np.c_[np.ones(X_val.shape[0]), X_val]
    val_loss = np.zeros(steps)
    train_loss = np.zeros(steps)

    for i in range(1, epochs + 1):
        perm = np.random.permutation(X.shape[0])
        X_perm = X[perm]
        y_perm = y[perm]
        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            X_batch, y_batch = X_perm[start:end], y_perm[start:end]

            gd = grad(X_batch, y_batch, theta)
            theta -= alpha * gd
            if i % (epochs // steps) == 0:
                val_loss[i // (epochs // steps) - 1] = loss(X_val, theta, y_val)
                train_loss[i // (epochs // steps) - 1] = loss(X, theta, y)
    return theta, val_loss, train_loss

def L2_analytic_linear_regression(X, y, X_val, y_val, intercept, lambdaa, loss):
    if intercept:
        X = np.c_[np.ones(X.shape[0]), X]

    I = np.eye(X.shape[1])
    if intercept:
        I[0, 0] = 0

    theta = np.linalg.pinv(X.T @ X + lambdaa * I) @ X.T @ y

    if intercept:
        X_val = np.c_[np.ones(X_val.shape[0]), X_val]

    val_loss = loss(X_val, theta, y_val)
    train_loss = loss(X, theta, y)
    return theta, [val_loss], [train_loss]

def L1_linear_regression(X, y, X_val, y_val, intercept, lambdaa, epochs, loss, steps):
    if intercept:
        X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros((X.shape[1], 1))
    val_loss = np.zeros(steps)
    train_loss = np.zeros(steps)

    if intercept:
        X_val = np.c_[np.ones(X_val.shape[0]), X_val]

    for i in range(1, epochs + 1):
        for j in range(0, theta.shape[0]):
            X_j = X[:, j].reshape((-1, 1))
            a = 2 * (X_j.T @ X_j)
            X_outj = np.delete(X, j, axis=1)
            theta_outj = np.delete(theta, j, axis=0)
            y_pred = np.matmul(X_outj, theta_outj)
            r = y - y_pred
            c = 2 * (X_j.T @ r)
            if j == 0 and intercept:
                theta[j] = c / a
                continue

            if c < -lambdaa:
                theta[j] = (c + lambdaa) / a
            elif c > lambdaa:
                theta[j] = (c - lambdaa) / a
            else:
                theta[j] = 0
        if i % (epochs // steps) == 0:
            val_loss[i // (epochs // steps) - 1] = loss(X_val, theta, y_val)
            train_loss[i // (epochs // steps) - 1] = loss(X, theta, y)

    return theta, val_loss, train_loss
