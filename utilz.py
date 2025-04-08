import numpy as np
import functools as ft
from warnings import catch_warnings

def divide_dataset(data, fractions_train_val_test=None):
    if fractions_train_val_test is None:
        fractions_train_val_test = [0.8, 0, 0.2]

    np.random.shuffle(data)
    train_ratio = fractions_train_val_test[0]
    val_ratio = fractions_train_val_test[1]
    # test_ratio - the rest
    n = len(data)
    n1 = int(n * train_ratio)
    n2 = int(n * val_ratio)
    return data[:n1], data[n1:n1+n2], data[n1+n2:]

def split_data(data):
    X = data[:, :-1]
    y = data[:, -1].reshape((-1, 1))
    return X, y

def run_linear_regression_model(model, loss, datasets, bias, use_validation):
    t = len(datasets)
    total_error = 0
    # not clean but works
    total_theta = np.zeros((datasets[0][0].shape[1] - 1 + int(bias), 1))
    total_loss_history = []

    for train, val, test in datasets:
        X_train, y_train = split_data(train)
        if use_validation:
            X_test, y_test = split_data(val)
        else:
            X_test, y_test = split_data(test)

        theta, losses = model(X_train, y_train, X_test, y_test)
        total_theta += theta
        total_loss_history.append(losses)

        X_test_plan = X_test if not bias else np.c_[np.ones(X_test.shape[0]), X_test]
        y_pred = X_test_plan @ theta

        error = loss(y_test, y_pred)
        total_error += error

    avg_error = total_error / t
    avg_theta = total_theta / t
    avg_loss_history = np.mean(total_loss_history, axis=0)

    return avg_theta, avg_error, avg_loss_history

def search_alphas(model, loss, datasets, alphas):
    for alpha in alphas:
        with catch_warnings(record=True):
            theta, error, _ = run_linear_regression_model(
                ft.partial(model, alpha=alpha, steps=1),
                loss,
                datasets,
                bias=True,
                use_validation=True)
            print(f"Alpha: {alpha}, loss {error}")
