import numpy as np


class GaussianProcess:
    """Small framework for Gaussian process exercises.

    Parameters
    ----------
    gamma : float
        Signal standard deviation (output scale).
    l : float
        Length scale of the squared exponential kernel.
    sigma_noise : float, optional
        Standard deviation of the observation noise. Default is 0.
    """

    def __init__(self, gamma: float, l: float, sigma_noise: float = 0.0):
        self.gamma = gamma
        self.l = l
        self.sigma_noise = sigma_noise

    def assemble_covariance_matrix(
        self, x: np.ndarray, x_prime: np.ndarray
    ) -> np.ndarray:
        """Assemble the covariance matrix using the squared exponential kernel.

        C(x, x') = gamma^2 * exp(-0.5 * (x - x')^2 / l^2)

        Parameters
        ----------
        x : array_like, shape (n,)
            First set of input points.
        x_prime : array_like, shape (m,)
            Second set of input points.

        Returns
        -------
        C : np.ndarray, shape (n, m)
            Covariance matrix.
        """
        x = np.asarray(x)
        x_prime = np.asarray(x_prime)
        diff = x[:, np.newaxis] - x_prime[np.newaxis, :]
        return self.gamma**2 * np.exp(-0.5 * diff**2 / self.l**2)

    def assemble_noisy_covariance_matrix(self, x: np.ndarray) -> np.ndarray:
        """Assemble the noisy covariance matrix.

        C_noisy(x, x') = C(x, x') + sigma_noise^2 * I

        Parameters
        ----------
        x : array_like, shape (n,)
            Input points.

        Returns
        -------
        C_noisy : np.ndarray, shape (n, n)
            Noisy covariance matrix with observation noise added to the diagonal.
        """
        x = np.asarray(x)
        C = self.assemble_covariance_matrix(x, x)
        return C + self.sigma_noise**2 * np.eye(len(x))
