import numpy as np
import functools as ft
from warnings import catch_warnings

def divide_dataset(data, fractions_train_val_test=None):
    if fractions_train_val_test is None:
        fractions_train_val_test = [0.6, 0.2, 0.2]

    np.random.seed(1882)
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

# use_test tells if we want to use test set for validation
# note that avg_loss is calculated on test set
def run_linear_regression_model(model, loss, datasets, standardize, use_test = False):
    t = len(datasets)
    total_loss = 0

    thetas = []
    total_loss_history = []
    total_training_history = []

    for train, val, test in datasets:
        X_train, y_train = split_data(train)
        X_val, y_val = split_data(val)
        X_test, y_test = split_data(test)

        if standardize:
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            X_test = (X_test - mean) / std

        if not use_test:
            theta, losses, train_losses = model(X_train, y_train, X_val, y_val)
        else:
            theta, losses, train_losses = model(X_train, y_train, X_test, y_test)

        thetas.append(theta)
        total_loss_history.append(losses)
        total_training_history.append(train_losses)

        if theta.shape[0] > X_train.shape[1]:
            X_test = np.c_[np.ones(X_test.shape[0]), X_test]

        test_loss = loss(X_test, theta, y_test)
        total_loss += test_loss

    avg_loss = total_loss / t
    avg_theta = np.mean(thetas, axis=0)
    avg_loss_history = np.mean(total_loss_history, axis=0)
    avg_training_history = np.mean(total_training_history, axis=0)

    return avg_theta, avg_loss, avg_loss_history, avg_training_history

def search_alpha(model, loss, datasets, alphas, standardize):
    for alpha in alphas:
        with catch_warnings(record=True):
            _, _, loss_history, _ = run_linear_regression_model(
                model=ft.partial(model, alpha=alpha, steps=1),
                loss=loss,
                datasets=datasets,
                standardize=standardize,
                use_test=False)
            print(f"Alpha: {alpha}, validation loss {loss_history[-1]}")

def search_lambdas(model, loss, datasets, lambdas, standardize):
    for lambdaa in lambdas:
        with catch_warnings(record=True):
            _, _, loss_history, _ = run_linear_regression_model(
                model=ft.partial(model, lambdaa=lambdaa),
                loss=loss,
                datasets=datasets,
                standardize=standardize,
                use_test=False)
            print(f"Lambda: {lambdaa}, validation loss {loss_history[-1]}")

def search_batch_size(model, loss, datasets, batch_sizes, standardize):
    for batch_size in batch_sizes:
        with catch_warnings(record=True):
            _, _, loss_history, _ = run_linear_regression_model(
                model=ft.partial(model, batch_size=batch_size, steps=1),
                loss=loss,
                datasets=datasets,
                standardize=standardize,
                use_test=False)
            print(f"Batch size: {batch_size}, validation loss {loss_history[-1]}")
