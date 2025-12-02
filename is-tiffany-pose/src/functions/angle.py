import numpy as np


def angle(A, B):
    """
    Calculates yaw and pitch efficiently directly from vector components.
    No rotation matrix or cross products needed.
    """
    v = B - A
    dx, dy, dz = v[0], v[1], v[2]

    yaw_rad = np.arctan2(dy, dx)
    yaw_deg = np.degrees(yaw_rad) % 360

    norm = np.sqrt(dx * dx + dy * dy + dz * dz)

    if norm < 1e-9:
        return 0.0, 0.0

    pitch_rad = np.arcsin(np.clip(dz / norm, -1.0, 1.0))
    pitch_deg = np.degrees(pitch_rad)

    return yaw_deg, pitch_deg
