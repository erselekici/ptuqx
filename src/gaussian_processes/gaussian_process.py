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
        self.training_inputs = None
        self.measurements = None
        self.inverse_covariance_matrix = None

    def fit_to_training_data(self, training_inputs: np.ndarray, measurements: np.ndarray):
        """
        Fit the Gaussian Process model to the training data.

        Args:
            training_inputs: The training input samples. (n_training_samples,)
            measurements: The training target values. (n_training_samples,)
            y_train: The training target values. (n_training_samples,)
        """
        self.training_inputs = training_inputs
        self.measurements = measurements
        covariance_matrix = self.kernel(training_inputs, training_inputs) + self.noise * np.eye(len(training_inputs))
        self.inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

    def predict(self, test_inputs: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Predict using the Gaussian Process model.

        Args:
            test_inputs: The test input samples. (n_test_samples,)

        Returns:
            mean: The mean predictions for the test inputs. (n_test_samples,)
            variance: The variance of the predictions for the test inputs. (n_test_samples,)
        """
        K_s = self.kernel(self.training_inputs, test_inputs)
        K_ss = self.kernel(test_inputs, test_inputs) + self.noise * np.eye(len(test_inputs))

        mean = K_s.T @ self.inverse_covariance_matrix @ self.measurements
        variance = K_ss - K_s.T @ self.inverse_covariance_matrix @ K_s

        return mean, np.diag(variance)