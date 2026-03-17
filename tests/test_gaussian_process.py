import numpy as np
import pytest
from gaussian_process import GaussianProcess


def test_covariance_matrix_shape():
    gp = GaussianProcess(gamma=1.0, l=1.0)
    x = np.array([0.0, 1.0, 2.0])
    x_prime = np.array([0.5, 1.5])
    C = gp.assemble_covariance_matrix(x, x_prime)
    assert C.shape == (3, 2)


def test_covariance_matrix_symmetry():
    gp = GaussianProcess(gamma=1.0, l=1.0)
    x = np.array([0.0, 1.0, 2.0])
    C = gp.assemble_covariance_matrix(x, x)
    np.testing.assert_allclose(C, C.T)


def test_covariance_matrix_diagonal():
    """Diagonal entries should equal gamma^2 (self-covariance)."""
    gamma = 2.0
    gp = GaussianProcess(gamma=gamma, l=1.0)
    x = np.array([0.0, 1.0, 2.0])
    C = gp.assemble_covariance_matrix(x, x)
    np.testing.assert_allclose(np.diag(C), gamma**2)


def test_covariance_matrix_values():
    """Verify a known value of the kernel."""
    gamma = 1.0
    l = 1.0
    gp = GaussianProcess(gamma=gamma, l=l)
    x = np.array([0.0])
    x_prime = np.array([1.0])
    C = gp.assemble_covariance_matrix(x, x_prime)
    expected = gamma**2 * np.exp(-0.5 * 1.0**2 / l**2)
    np.testing.assert_allclose(C[0, 0], expected)


def test_covariance_matrix_gamma_scaling():
    """Covariance should scale with gamma^2."""
    x = np.array([0.0, 1.0])
    gp1 = GaussianProcess(gamma=1.0, l=1.0)
    gp2 = GaussianProcess(gamma=2.0, l=1.0)
    C1 = gp1.assemble_covariance_matrix(x, x)
    C2 = gp2.assemble_covariance_matrix(x, x)
    np.testing.assert_allclose(C2, 4.0 * C1)


def test_noisy_covariance_matrix_shape():
    gp = GaussianProcess(gamma=1.0, l=1.0, sigma_noise=0.1)
    x = np.array([0.0, 1.0, 2.0])
    C_noisy = gp.assemble_noisy_covariance_matrix(x)
    assert C_noisy.shape == (3, 3)


def test_noisy_covariance_matrix_diagonal():
    """Diagonal of noisy matrix should be gamma^2 + sigma_noise^2."""
    gamma = 1.0
    sigma_noise = 0.5
    gp = GaussianProcess(gamma=gamma, l=1.0, sigma_noise=sigma_noise)
    x = np.array([0.0, 1.0, 2.0])
    C_noisy = gp.assemble_noisy_covariance_matrix(x)
    expected_diag = gamma**2 + sigma_noise**2
    np.testing.assert_allclose(np.diag(C_noisy), expected_diag)


def test_noisy_covariance_matrix_zero_noise():
    """With zero noise, noisy matrix should equal clean covariance matrix."""
    gp = GaussianProcess(gamma=1.0, l=1.0, sigma_noise=0.0)
    x = np.array([0.0, 1.0, 2.0])
    C = gp.assemble_covariance_matrix(x, x)
    C_noisy = gp.assemble_noisy_covariance_matrix(x)
    np.testing.assert_allclose(C_noisy, C)


def test_noisy_covariance_matrix_off_diagonal():
    """Off-diagonal entries should be unchanged by noise."""
    gp = GaussianProcess(gamma=1.0, l=1.0, sigma_noise=0.5)
    x = np.array([0.0, 1.0])
    C = gp.assemble_covariance_matrix(x, x)
    C_noisy = gp.assemble_noisy_covariance_matrix(x)
    np.testing.assert_allclose(C_noisy[0, 1], C[0, 1])
    np.testing.assert_allclose(C_noisy[1, 0], C[1, 0])
