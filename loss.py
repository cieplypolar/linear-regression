import numpy as np

def ols(y, y_pred):
    return np.mean((y - y_pred) ** 2) * 0.5