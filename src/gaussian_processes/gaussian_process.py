"""Module for Gaussian Processes."""

import numpy as np


class GaussianProcess:
    """Gaussian Process regression model."""

    def __init__(self, kernel: callable, noise: float):
        """
        Initialize the Gaussian Process.

        Args:
            kernel: A callable that takes two arrays and returns a covariance matrix.
            noise: The noise level (variance) of the observations.
        """
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the Gaussian Process model to the training data.

        Args:
            X_train: The training input samples. (n_samples x n_features)
            y_train: The training target values. (n_samples,)
        """
        self.X_train = X_train
        self.y_train = y_train
        K = self.kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Predict using the Gaussian Process model.

        Args:
            X_test: The test input samples. (n_samples x n_features)

        Returns:
            mean: The mean predictions for the test inputs. (n_samples,)
            variance: The variance of the predictions for the test inputs. (n_samples,)
        """
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test) + self.noise * np.eye(len(X_test))

        mean = K_s.T @ self.K_inv @ self.y_train
        variance = K_ss - K_s.T @ self.K_inv @ K_s

        return mean, np.diag(variance)