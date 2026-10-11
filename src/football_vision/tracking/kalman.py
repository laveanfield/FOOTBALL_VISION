"""Kalman filter for tracking with a constant-velocity motion model.

State ``x_k`` evolves as ``x_k = F x_{k-1} + w`` with process noise ``w ~ N(0, Q)``
and is observed as ``z_k = M x_k + v`` with measurement noise ``v ~ N(0, R)``.
The matrix ``M`` is used for the measurement model to avoid a clash with the
homography ``H``.

Predict:
    ``x^- = F x``, ``P^- = F P F^T + Q``
Update:
    ``S = M P^- M^T + R``, ``K = P^- M^T S^-1``,
    ``x = x^- + K (z - M x^-)``, ``P = (I - K M) P^-``
The gain ``K`` minimizes the trace of the posterior covariance (Appendix A5).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def constant_velocity_matrices(
    dt: float,
    process_std: float,
    measurement_std: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Builds the matrices of a 2D constant-velocity model in pitch space.

    The state is ``[X, Y, vX, vY]`` and only the position is measured. The process
    noise is a white-noise acceleration model with standard deviation ``process_std``.

    Args:
        dt: Time step in seconds (``stride / fps``).
        process_std: Standard deviation of the acceleration noise in m/s^2.
        measurement_std: Standard deviation of the position measurement in meters.

    Returns:
        Tuple ``(F, M, Q, R)`` with shapes ``(4, 4)``, ``(2, 4)``, ``(4, 4)``
        and ``(2, 2)``.
    """
    f = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    m = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    block = (
        np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]) * process_std**2
    )
    q = np.zeros((4, 4))
    q[np.ix_([0, 2], [0, 2])] = block
    q[np.ix_([1, 3], [1, 3])] = block
    r = np.eye(2) * measurement_std**2
    return f, m, q, r


class KalmanFilter:
    """Linear Kalman filter with a configurable motion and measurement model.

    Attributes:
        f: State transition matrix of shape ``(dim_x, dim_x)``.
        m: Measurement matrix of shape ``(dim_z, dim_x)``.
        q: Process noise covariance of shape ``(dim_x, dim_x)``.
        r: Measurement noise covariance of shape ``(dim_z, dim_z)``.
        x: Current state estimate of shape ``(dim_x,)``.
        p: Current state covariance of shape ``(dim_x, dim_x)``.
    """

    def __init__(
        self,
        f: npt.ArrayLike,
        m: npt.ArrayLike,
        q: npt.ArrayLike,
        r: npt.ArrayLike,
        x0: npt.ArrayLike,
        p0: npt.ArrayLike,
    ) -> None:
        """Creates a filter from its model and initial state.

        Args:
            f: State transition matrix.
            m: Measurement matrix.
            q: Process noise covariance.
            r: Measurement noise covariance.
            x0: Initial state, shape ``(dim_x,)``.
            p0: Initial state covariance.
        """
        self.f = np.asarray(f, dtype=np.float64)
        self.m = np.asarray(m, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)
        self.r = np.asarray(r, dtype=np.float64)
        self.x = np.asarray(x0, dtype=np.float64).reshape(-1)
        self.p = np.asarray(p0, dtype=np.float64)

    def predict(self) -> npt.NDArray[np.float64]:
        """Propagates the state and covariance one step with the motion model.

        Returns:
            The predicted state of shape ``(dim_x,)``; ``x`` and ``p`` are updated.

        Raises:
            NotImplementedError: Stub; to be implemented by member C.
        """
        # TODO(C): x = F x ; P = F P F^T + Q  (see the module docstring).
        raise NotImplementedError("TODO(C): implement KalmanFilter.predict")

    def update(self, z: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Corrects the prediction with a measurement.

        Args:
            z: Measurement of shape ``(dim_z,)``.

        Returns:
            The corrected state of shape ``(dim_x,)``; ``x`` and ``p`` are updated.

        Raises:
            NotImplementedError: Stub; to be implemented by member C.
        """
        # TODO(C): S = M P M^T + R ; K = P M^T S^-1 ; x += K (z - M x) ;
        #   P = (I - K M) P. Prefer np.linalg.solve over an explicit inverse.
        raise NotImplementedError("TODO(C): implement KalmanFilter.update")

    def gating_distance(self, measurements: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Computes squared Mahalanobis distances from the prediction to measurements.

        Under the Gaussian model the squared distance follows a chi-square law, so a
        gate at the 95% quantile (5.99 for 2 degrees of freedom) rejects implausible
        associations.

        Args:
            measurements: Array of shape ``(N, dim_z)``.

        Returns:
            Array of shape ``(N,)`` with squared Mahalanobis distances.

        Raises:
            NotImplementedError: Stub; to be implemented by member C.
        """
        # TODO(C): d^2 = (z - M x)^T S^-1 (z - M x) with S = M P M^T + R.
        raise NotImplementedError("TODO(C): implement KalmanFilter.gating_distance")
