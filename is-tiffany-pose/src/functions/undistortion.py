import cv2
import numpy as np


def undistortPoints(parameters: dict, points: np.ndarray):
    """
    Undistorts image points using camera calibration parameters.

    Args:
        parameters (dict): Dictionary containing camera calibration data:
            - 'K': Camera intrinsic matrix.
            - 'rt': Rotation-translation matrix.
            - 'dist': Distortion coefficients.
        points (np.ndarray): Array of image points to undistort, shape (N,2) or (1,N,2).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Projection matrix (3x4) combining intrinsic and extrinsic parameters.
            - Undistorted image points.
    """
    mtxK = parameters["K"]
    Rtcw = parameters["rt"]
    dis = parameters["dist"]
    newK = parameters["nK"]

    undistorted = cv2.undistortPoints(points, mtxK, dis, None, newK).squeeze(axis=0)
    mtxP = newK @ Rtcw

    return mtxP, undistorted


def point2world(parameters: dict, points: dict) -> np.ndarray:
    """
    Computes 3D world coordinates from 2D image points across multiple cameras.

    Args:
        parameters (dict): Camera calibration parameters for each camera.
        points (dict): Dictionary of 2D image points per camera {camera_id: points}.

    Returns:
        np.ndarray: 3D coordinates of the reconstructed point in world space.
    """
    mtxA = []
    for cam_id in points.keys():
        mtxP, unds = undistortPoints(parameters[cam_id], points[cam_id])
        unds = unds[0]
        u = np.array([[int(unds[0]), int(unds[1]), 1]]).T
        zeros = np.zeros((3, 1))
        mtx = np.array(mtxP)

    # Construct matrix with extrinsic info for each camera
    for i in points.keys():
        if i == cam_id:
            mtx = np.hstack((mtx, -u))
        else:
            mtx = np.hstack((mtx, zeros))

    # Stack to form the full A matrix
    if len(mtxA) == 0:
        mtxA = mtx
    else:
        mtxA = np.vstack((mtxA, mtx))

    # Solve using SVD
    _, _, V_transpose = np.linalg.svd(mtxA)
    mtxV = V_transpose[-1]
    mtxV = mtxV / mtxV[3]  # normalize homogeneous coordinate
    Xw = mtxV[:3]

    return Xw
