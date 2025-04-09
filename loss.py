import numpy as np

def ols(X, theta, y):
    y_pred = X @ theta
    error = np.mean((y_pred - y) ** 2) * 0.5
    return error

def l2_loss(X, theta, y, lambdaa):
    error = ols(X, theta, y)
    norm = np.sum(theta[1:] ** 2) * lambdaa
    return error + norm
