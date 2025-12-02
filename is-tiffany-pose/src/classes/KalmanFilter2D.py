import numpy as np


class KalmanFilter2D:
    """Kalman filter for 2D position and velocity tracking.
    Otimizado para reutilização de memória e cálculo matricial eficiente.
    """

    def __init__(
        self, dt=0.1, process_var=1e-8, meas_var=50.0, vel_scale=10.0, gate=None
    ):
        self.dt = dt

        self.F = np.eye(4, dtype=float)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.H = np.zeros((2, 4), dtype=float)
        self.H[0, 0] = 1
        self.H[1, 1] = 1

        self.base_process_var = float(process_var)
        self.base_meas_var = float(meas_var)
        self.vel_scale = float(vel_scale)
        self.gate = gate

        self._update_QR()

        self.I = np.eye(4, dtype=float)

        self.P = np.eye(4, dtype=float)
        self.x = np.zeros(4, dtype=float)
        self.initialized = False

    def reset(self):
        """Reinicia o estado do filtro mantendo as configurações."""
        self.x.fill(0.0)
        self.P[:] = np.eye(4) * 100.0
        self.initialized = False

    def _update_QR(self):
        q_pos = self.base_process_var
        q_vel = self.base_process_var * self.vel_scale
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.R = np.eye(2) * self.base_meas_var

    def set_dt(self, dt):
        dt = np.clip(dt, 1e-3, 0.1)
        self.dt = dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt

    def initialize(self, pos):
        self.x[:2] = pos
        self.x[2:] = 0.0
        self.P[:] = np.eye(4) * self.base_meas_var
        self.initialized = True

    def predict(self):
        # x = F @ x
        self.x = self.F @ self.x
        # P = F @ P @ F.T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        z = np.asarray(measurement)

        if not self.initialized:
            self.initialize(z)
            return self.x[:2]

        y = z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R

        try:
            invS = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x[:2]

        if self.gate is not None:
            mahal = float(y.T @ invS @ y)
            if mahal > self.gate:
                self.P += self.Q * 0.1
                return self.x[:2]

        K = self.P @ self.H.T @ invS

        self.x = self.x + K @ y

        self.P = (self.I - K @ self.H) @ self.P

        return self.x[:2]
