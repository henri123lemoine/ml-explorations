import numpy as np

from src.utils.data_processing import add_bias_term

np.random.seed(42)


def gradient_descent(
    model, X, Y, predict_func, learning_rate, epochs, test_func, test_interval, test_start
):
    test_results = []
    X = add_bias_term(X)
    num_samples, num_features = X.shape

    # Initialize weights using normal
    if len(Y.shape) == 1:
        model.weights = np.random.normal(0, 0.0005, num_features)
    else:
        model.weights = np.random.normal(0, 0.0005, (num_features, Y.shape[1]))

    # Gradient Descent
    for epoch in range(epochs):
        predictions = predict_func(model.weights, X)
        # Compute gradients
        dw = (1 / num_samples) * np.dot(X.T, (predictions - Y))
        # Update parameters
        model.weights -= learning_rate * dw

        if test_func is not None and epoch % test_interval == 0 and epoch >= test_start:
            test_results.append(test_func(model))

    return test_results


def stochastic_gradient_descent(
    model,
    X,
    Y,
    predict_func,
    learning_rate,
    epochs,
    batch_size,
    test_func,
    test_interval,
    test_start,
):
    test_results = []
    X = add_bias_term(X)
    num_samples, num_features = X.shape

    # Calculate the adjusted number of epochs
    adjusted_epochs = epochs * (num_samples // batch_size)

    # Initialize weights
    if len(Y.shape) == 1:
        model.weights = np.random.normal(0, 0.0005, num_features)
    else:
        model.weights = np.random.normal(0, 0.0005, (num_features, Y.shape[1]))

    # Stochastic Gradient Descent
    for epoch in range(adjusted_epochs):
        # Shuffle data
        shuffle = np.random.permutation(len(Y))
        X_shuffled = X[shuffle]
        Y_shuffled = Y[shuffle]

        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            Y_batch = Y_shuffled[i : i + batch_size]

            predictions = predict_func(model.weights, X_batch)
            # Compute gradients
            dw = (1 / batch_size) * X_batch.T @ (predictions - Y_batch)
            # Update parameters
            model.weights -= learning_rate * dw

        if test_func is not None and epoch % test_interval == 0 and epoch >= test_start:
            test_results.append(test_func(model))

    return test_results
