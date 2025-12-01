import numpy as np


class KalmanFilter2D:
    """Kalman filter for 2D position and velocity tracking.

    Notes:
    - State vector: [x, y, vx, vy]
    - Model: constant velocity (CV)
    - This version adds convenient tuning helpers (process/measurement variance,
      velocity scaling), an inertia helper, and optional Mahalanobis gating to
      reject large outliers when measurements are imprecise.
    """

    def __init__(
        self, dt=0.1, process_var=1e-8, meas_var=50.0, vel_scale=10.0, gate=None
    ):
        # time step
        self.dt = dt

        # state transition matrix (CV model)
        self.F = np.eye(4)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # measurement matrix: measure only positions (x, y)
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1
        self.H[1, 1] = 1

        # store base params so we can safely reconfigure
        self.base_process_var = float(process_var)
        self.base_meas_var = float(meas_var)
        self.vel_scale = float(vel_scale)

        # gating threshold (Mahalanobis distance). If None, gating is disabled.
        # Suggested values (chi-square for 2 dof): 95% -> 5.991, 99% -> 9.210
        self.gate = gate

        # initialize Q, R from base params
        self._update_QR()

        # state covariance and state vector
        self.P = np.eye(4)
        self.x = np.zeros(4)
        self.initialized = False

    def _update_QR(self):
        """(Re)compute process (Q) and measurement (R) covariances from base params."""
        q_pos = self.base_process_var
        q_vel = self.base_process_var * self.vel_scale
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.R = np.eye(2) * self.base_meas_var

    def set_dt(self, dt):
        """Update time step (clip to reasonable range)."""
        dt = np.clip(dt, 1e-3, 0.1)
        self.dt = dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt

    # Tuning helpers
    def set_process_var(self, process_var):
        """Set process noise base variance (affects Q)."""
        self.base_process_var = float(process_var)
        self._update_QR()

    def set_meas_var(self, meas_var):
        """Set measurement noise variance (affects R)."""
        self.base_meas_var = float(meas_var)
        self._update_QR()

    def set_velocity_scale(self, vel_scale):
        """Set velocity variance scale relative to position process variance."""
        self.vel_scale = float(vel_scale)
        self._update_QR()

    def set_initial_uncertainty(self, value):
        """Set initial state covariance P = I * value."""
        self.P = np.eye(4) * float(value)

    def set_inertia(self, inertia):
        """Adjust 'inertia' globally.

        inertia > 1 -> increase inertia (less responsive to measurements)
        0 < inertia < 1 -> decrease inertia (more responsive)

        Implementation: multiply measurement variance by inertia and divide process variance
        by inertia. This reduces Kalman gain, making the filter slower to react to noisy
        measurements.
        """
        if inertia <= 0:
            raise ValueError("inertia must be > 0")
        self.base_meas_var *= float(inertia)
        self.base_process_var /= float(inertia)
        self._update_QR()

    def set_gate(self, gate):
        """Set Mahalanobis gating threshold. None disables gating."""
        if gate is not None and gate <= 0:
            raise ValueError("gate must be positive or None")
        self.gate = gate

    def initialize(self, pos):
        """Initialize state with a position (vx,vy set to zero)."""
        self.x[:2] = pos
        self.x[2:] = 0.0
        self.initialized = True

    def predict(self):
        """Predict step: x = F x, P = F P F^T + Q"""
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, measurement):
        """Update step with optional Mahalanobis gating.

        If gating is enabled and the measurement is an outlier (Mahalanobis distance >
        gate), the measurement will be ignored and the predicted state returned.
        """
        z = np.asarray(measurement)
        if not self.initialized:
            self.initialize(z)
            return self.x[:2]

        # Innovation covariance and residual
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        y = z - self.H.dot(self.x)

        # Mahalanobis gating (optional)
        if self.gate is not None:
            try:
                invS = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                invS = np.linalg.pinv(S)
            mahal = float(y.T.dot(invS).dot(y))
            if mahal > self.gate:
                # measurement considered an outlier -> skip update
                # Inflate P slightly to reflect ignored measurement (optional)
                self.P = self.P + self.Q * 0.1
                return self.x[:2]

        # Standard Kalman update
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        I = np.eye(4)
        self.P = (I - K.dot(self.H)).dot(self.P)
        return self.x[:2]
