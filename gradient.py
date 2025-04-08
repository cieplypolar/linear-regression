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