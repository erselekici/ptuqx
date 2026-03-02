"""Testing module for Gaussian processes."""

import numpy as np
import pytest

from src.gaussian_processes.gaussian_process import GaussianProcess


@pytest.fixture(name="kernel")
def fixture_kernel():
    """A simple kernel function for testing."""
    def kernel(x, y):
        return np.exp(-0.5 * (x[:, None] - y[None, :]) ** 2)
    return kernel


@pytest.fixture(name="training_data")
def fixture_training_data():
    """Simple training data for testing."""
    training_inputs = np.array([0.0, 1.0, 2.0])
    measurements = np.array([1.0, 2.0, 3.0])
    return training_inputs, measurements


@pytest.mark.parametrize("noise, reference_inverse_covariance_matrix", [
    (0, np.array([[ 1.82958397, -1.51793414,  0.67306633],
                   [-1.51793414,  2.84134719, -1.51793414],
                   [ 0.67306633, -1.51793414,  1.82958397]])),
    (0.5, np.array([[ 0.80313667, -0.35320004,  0.07035595],
                     [-0.35320004,  0.9523022 , -0.35320004],
                     [ 0.07035595, -0.35320004,  0.80313667]])),
])
def test_fit_to_training_data(kernel, noise, training_data, reference_inverse_covariance_matrix):
    """Test fitting the Gaussian Process to training data."""
    training_inputs, measurements = training_data
    gp = GaussianProcess(kernel, noise)
    gp.fit_to_training_data(training_inputs, measurements)

    assert np.allclose(gp.inverse_covariance_matrix, reference_inverse_covariance_matrix)