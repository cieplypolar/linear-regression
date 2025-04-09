import numpy as np

def finite_difference_gradient(X, y, theta, epsilon):
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        y_pred_plus = X @ theta_plus
        y_pred_minus = X @ theta_minus

        loss_plus = np.mean((y - y_pred_plus) ** 2) * 0.5
        loss_minus = np.mean((y - y_pred_minus) ** 2) * 0.5

        grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
    return grad

def GD_OLS(X, y, theta):
    y_pred = X @ theta
    er = y_pred - y
    return np.mean(X * er, axis=0).reshape((-1, 1))

def GD_OLS_L2(X, y, theta, lambdaa):
    y_pred = X @ theta
    er = y_pred - y
    norm = theta.copy()
    norm[0] = 0
    norm = lambdaa * 2 * norm
    return np.sum(X * er, axis=0).reshape((-1, 1)) + norm