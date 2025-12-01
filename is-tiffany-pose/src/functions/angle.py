import numpy as np


def normalize(v, eps=1e-9):
    """Normalize a vector, safely handling near-zero norms."""
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def angle(A, B, world_up=np.array([0.0, 0.0, 1.0])):
    """
    Computes yaw and pitch (in degrees) from two 3D points on the robot.

    Args:
        A (np.ndarray): 3D coordinates of the robot center.
        B (np.ndarray): 3D coordinates of the robot front.
        world_up (np.ndarray): The world's up direction (default: z-up).

    Returns:
        (float, float): yaw and pitch in degrees, yaw in range [0, 360) and pitch [-90, 90].
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    u = normalize(np.asarray(world_up, dtype=float))

    # Robot's forward axis
    x = normalize(B - A)

    # If the robot's forward vector is almost parallel to world_up,
    # use a small offset to avoid numerical instability (gimbal lock)
    y = np.cross(x, u)
    if np.linalg.norm(y) < 1e-6:
        # Apply a small perturbation to world_up
        u = normalize(u + np.array([1e-3, 0, 0]))
        y = np.cross(x, u)
    y = normalize(y)
    z = np.cross(x, y)

    # Rotation matrix with columns = [x, y, z]
    R = np.column_stack((x, y, z))

    # Extract yaw (psi) and pitch (theta) using Z-Y-X convention
    psi = np.arctan2(R[1, 0], R[0, 0])  # yaw
    theta = np.arcsin(-R[2, 0])  # pitch

    # Convert yaw to range [0, 360)
    yaw_deg = np.degrees(psi) % 360
    pitch_deg = -np.degrees(theta)

    return yaw_deg, pitch_deg
