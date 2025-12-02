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
    mtxK = np.asarray(parameters.get("K"))
    Rtcw = parameters.get("rt")
    dis = parameters.get("dist", None)
    newK = np.asarray(parameters.get("nK", mtxK))

    pts = np.asarray(points)
    # prepare input for cv2.undistortPoints: shape (N,1,2)
    if pts.ndim == 1 and pts.size == 2:
        pts_cv = pts.reshape(1, 1, 2).astype(np.float32)
    elif pts.ndim == 2 and pts.shape[1] == 2:
        pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
    elif pts.ndim == 3:
        pts_cv = pts.astype(np.float32)
    else:
        raise ValueError("Formato de pontos não suportado para undistortPoints")

    undistorted = cv2.undistortPoints(pts_cv, mtxK, dis, None, newK)
    undistorted = undistorted.reshape(-1, 2)

    # Rtcw pode ser 3x4 (Rt) ou 3x3 (R) com t em parameters['t']
    Rtcw_arr = np.asarray(Rtcw)
    if Rtcw_arr.shape == (3, 4):
        Rt = Rtcw_arr
    elif Rtcw_arr.shape == (3, 3):
        t = np.asarray(parameters.get('t', np.zeros(3))).reshape(3, 1)
        Rt = np.hstack((Rtcw_arr, t))
    else:
        raise ValueError("Parâmetro 'rt' deve ser 3x4 ou 3x3")

    mtxP = newK @ Rt
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
    # points: dict {cam_id: (u,v) or array-like}
    Ps = []
    xs = []
    for cam_id, pt in points.items():
        if cam_id not in parameters:
            raise KeyError(f"Parâmetros da câmera {cam_id} não fornecidos")
        P, und = undistortPoints(parameters[cam_id], pt)
        Ps.append(np.asarray(P))
        # und is (N,2); take the first point (we reconstruct one 3D point)
        if und.size == 0:
            raise ValueError(f"Nenhum ponto undistorted para câmera {cam_id}")
        u, v = float(und.reshape(-1)[0]), float(und.reshape(-1)[1])
        xs.append((u, v))

    if len(Ps) < 2:
        raise ValueError("É necessário pelo menos duas câmeras para triangulação")

    # Build linear system A X = 0 (DLT)
    A_rows = []
    for P, (u, v) in zip(Ps, xs):
        A_rows.append(u * P[2, :] - P[0, :])
        A_rows.append(v * P[2, :] - P[1, :])
    A = np.vstack(A_rows)

    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        raise ValueError("Triangulação degenerada (ponto no infinito)")
    Xw = (Xh[:3] / Xh[3]).astype(float)
    return Xw
