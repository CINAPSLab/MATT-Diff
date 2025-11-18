import numpy as np

class KalmanFilter:
    def __init__(self, init_pos, process_noise_sigmas=None, dt=0.1, init_P=None, Q=None, R=None):
        """
        Initializes the Kalman Filter.
        Args:
            init_pos (np.ndarray): Initial [x, y] position.
            process_noise_sigmas (list or tuple): Standard deviations of the process noise, e.g., [sigma_x, sigma_y].
            dt (float): Time step (unused for random walk).
            init_P (np.ndarray or None): Optional initial covariance (2x2).
            Q (np.ndarray or None): Optional process noise covariance (2x2).
            R (np.ndarray or None): Optional observation noise covariance (2x2).
        """
        self.x = np.array([init_pos[0], init_pos[1]], dtype=float)
        # Initial uncertainty
        if init_P is None:
            self.P = np.eye(2, dtype=float) * 25.0
        else:
            self.P = np.array(init_P, dtype=float).reshape(2, 2)

        self.F = np.eye(2, dtype=float)  # random walk

        # Process noise Q (variance per step; random-walk)
        if Q is not None:
            self.Q = np.array(Q, dtype=float).reshape(2, 2)
        else:
            if process_noise_sigmas is None:
                # Default to Q = diag([90, 40]) so that σy reaches ~200 in ~900–1000 steps
                self.Q = np.array([[90.0, 0.0],
                                   [0.0, 40.0]], dtype=float)
            else:
                sigma_x_sq = float(process_noise_sigmas[0]) ** 2
                sigma_y_sq = float(process_noise_sigmas[1]) ** 2
                self.Q = np.array([[sigma_x_sq, 0.0],
                                   [0.0,        sigma_y_sq]], dtype=float)

        self.H = np.eye(2, dtype=float)

        # Observation noise
        if R is not None:
            self.R = np.array(R, dtype=float).reshape(2, 2)
        else:
            self.R = np.eye(2, dtype=float) * 0.25

        # Tracking for "seen" and last update step (for recent_mask)
        self._seen = False
        self._last_update_step = -1

    def predict(self):
        """Predict the next state and covariance."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """Update the state estimate with a new measurement z."""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P
        self._seen = True
        # _last_update_step is set by the coordinator (wall-clock step), keep here for compatibility

    def innovation(self, z):
        """
        Returns the innovation y and its covariance S for a measurement z.
        Useful for Mahalanobis gating.
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def sigma_xy(self):
        """Return (sigma_x, sigma_y) = sqrt of diagonal of P clipped to non-negative."""
        diag = np.maximum(np.diag(self.P[:2, :2]), 0.0)
        return float(np.sqrt(diag[0])), float(np.sqrt(diag[1]))

    def mark_seen(self, step: int):
        """Mark this KF as updated at a given env step."""
        self._seen = True
        self._last_update_step = int(step)


def cov_ellipse(mean, cov, n_std=2.0):
    """
    Calculates parameters for drawing a covariance ellipse with cv2.ellipse.

    PROBLEM 2 SOLUTION: This function is now documented to "describe" the ellipse.
    The logic itself was correct, but the KF parameters were making the ellipse
    too small to be seen. The default n_std is also changed to 2.0.

    The ellipse represents the uncertainty of the Kalman Filter's estimate.
    - n_std=1.0: 1-sigma ellipse, encloses ~68% of the probability mass.
    - n_std=2.0: 2-sigma ellipse, encloses ~95% (a common choice for visualization).
    - n_std=3.0: 3-sigma ellipse, encloses ~99.7%.

    Args:
        mean (np.ndarray): The center of the ellipse [x, y].
        cov (np.ndarray): The 2x2 covariance matrix for the x, y state.
        n_std (float): The number of standard deviations to scale the ellipse axes.

    Returns:
        tuple: (center, axes, angle) for use with cv2.ellipse.
            - center (tuple): (int_x, int_y)
            - axes (tuple): (int_half_major_axis, int_half_minor_axis)
            - angle (float): The rotation angle in degrees.
    """
    # Eigenvalue decomposition of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues/vectors in descending order (major axis first)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Angle of the major axis is the angle of the first eigenvector
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Width/height are 2 * n_std * sqrt(eigenvalue)
    # Eigenvalues are variances, so sqrt gives std dev.
    width, height = 2 * n_std * np.sqrt(eigvals)

    # cv2.ellipse requires integer center and half-axis lengths
    center = tuple(np.round(mean).astype(int))
    axes = tuple(np.round([width / 2, height / 2]).astype(int))

    return center, axes, angle