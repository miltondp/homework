"""
This modules has been implemented in PyCharm using Github Copilot, a tool that
I usually employ to write code. I used Github Copilot to write simple code, and
never to solve the assigments. I left several comments explaining all the steps.
"""

import numpy as np


class LinearRegression:
    """
    Implements a linear regression model using the closed form/analytical
    solution.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        # initialize the weights and bias to None, meaning that the model has
        # not been fit yet
        self.w = None
        self.b = None

    @staticmethod
    def _add_constant(X: np.ndarray) -> np.ndarray:
        """
        Add a column of ones to the input matrix X. This is used to avoid having
        to add a bias term to the model.
        """
        return np.c_[np.ones(X.shape[0]), X]

    @staticmethod
    def _split_w_and_b(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the weights and bias from the parameter vector w.
        """
        return w[1:], w[0]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the model to the data using the closed form/analytical solution:

            w = (X^T @ X)^-1 @ X^T @ y

        where ^T represents the transpose and @ represents the matrix product.

        Args:
            X: The training data with shape (n_samples, n_features).
            y: The labels with shape (n_samples,).

        Returns:
            The same instance (LinearRegresion) with the fitted parameters.
        """
        # add a column of ones to X to avoid having to add a bias term
        X = LinearRegression._add_constant(X)

        # get inverse of covariance matrix; here I use the pseudo-inverse to
        # avoid having to check if the matrix is invertible
        XX_inv = np.linalg.pinv(X.T @ X)
        w = XX_inv @ X.T @ y

        # get weights and bias (first element of w, the rest are the weights)
        self.w, self.b = LinearRegression._split_w_and_b(w)

        # return self to allow chaining (such as model.fit(X, y).predict(X))
        return self

    @staticmethod
    def _predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input and weights. It assumes that the
        input matrix X has a column of ones as the first column, and that the
        bias term is the first element of the weights vector.
        """
        return X @ w

    def predict(self, X: np.ndarray) -> np.array:
        """
        Predict the output for the given input.
        """
        # check that model has been fit (self.w and self.b are not None)
        if self.w is None or self.b is None:
            raise ValueError("Model has not been fit yet.")

        # return y_hat (the predicted output given X)
        return self._predict(LinearRegression._add_constant(X), np.r_[self.b, self.w])


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.

    Args:
        adagrad: Whether to use Adagrad or not. See below for details on my
            tests using main.py (California housing data).
    """

    def __init__(self, adagrad: bool = True):
        super(GradientDescentLinearRegression, self).__init__()
        self.adagrad = adagrad

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.02, epochs: int = 1000
    ) -> "GradientDescentLinearRegression":
        """
        Fit the model to the data using gradient descent.

        Args:
            X: The training data with shape (n_samples, n_features).
            y: The labels with shape (n_samples,).
            lr: The learning rate.
            epochs: The number of epochs.

        Returns:
            The same instance (GradientDescentLinearRegression) with the fitted
            parameters.
        """
        X = LinearRegression._add_constant(X)
        w_ = np.zeros(X.shape[1])
        s = np.zeros(X.shape[1])

        for i in range(epochs):
            # get the predicted output
            y_hat = self._predict(X, w_)
            errors = y_hat - y

            # compute the gradient of the loss function
            grad = (2 / X.shape[0]) * (errors @ X)

            if self.adagrad:
                # here I use an adaptive learning rate with AdaGrad.
                # I found that it works better for the dataset in main.py
                # (California housing) than onlly using a fixed learning rate.
                # The best fixed learning rate was around lr=1e-7 with mse~1.44
                # (greater than this produced instabilities in the convergence).
                # On the other hand, using adagrad=True, it allowed to use
                # larger learning rates (around lr=0.02) and get a final mse
                # of 0.85. This means that AdaGrad allows to make larger updates
                # on the parameter at the beginning, and adaptively adjust the
                # learning rate to smaller values as the model converges.
                s += grad**2
                w_ -= lr * grad / (np.sqrt(s) + 1e-8)

            else:
                # update the weights
                w_ -= lr * grad

        self.w, self.b = LinearRegression._split_w_and_b(w_)

        return self
