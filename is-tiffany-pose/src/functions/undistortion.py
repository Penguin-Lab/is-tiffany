import cv2
import numpy as np


def precompute_projections(parameters: dict):
    """
    Gera um cache das matrizes de projeção (P) para evitar conta repetida.
    Call this ONCE inside your Threading.__init__.
    """
    projections = {}

    for cam_id, params in parameters.items():
        mtxK = np.asarray(params["K"], dtype=np.float64)
        nK = np.asarray(params.get("nK", mtxK), dtype=np.float64)

        Rtcw = np.asarray(params["rt"], dtype=np.float64)
        if Rtcw.shape == (3, 3):
            t = np.asarray(params.get("t", np.zeros(3)), dtype=np.float64).reshape(3, 1)
            Rt = np.hstack((Rtcw, t))
        else:
            Rt = Rtcw

        P = nK @ Rt

        dist = np.asarray(params.get("dist", []), dtype=np.float64)
        projections[cam_id] = {"P": P, "K": mtxK, "dist": dist, "nK": nK}

    return projections


def point2world(projections_cache: dict, points: dict) -> np.ndarray:
    """
    Triangulação otimizada usando cache pré-calculado.

    Args:
        projections_cache: Retorno de precompute_projections()
        points: Dict {cam_id: (x, y)}
    """
    num_cams = len(points)
    A = np.zeros((num_cams * 2, 4), dtype=np.float64)

    row_idx = 0

    for cam_id, pt in points.items():
        cache = projections_cache.get(cam_id)
        if cache is None:
            continue

        P = cache["P"]
        K = cache["K"]
        dist = cache["dist"]
        nK = cache["nK"]

        pt_cv = np.array([[[pt[0], pt[1]]]], dtype=np.float64)

        undistorted = cv2.undistortPoints(pt_cv, K, dist, None, nK)

        u = undistorted[0, 0, 0]
        v = undistorted[0, 0, 1]

        A[row_idx, :] = u * P[2] - P[0]
        A[row_idx + 1, :] = v * P[2] - P[1]

        row_idx += 2

    _, _, Vt = np.linalg.svd(A)

    X = Vt[-1]

    w = X[3]
    if abs(w) < 1e-9:
        return np.array([0.0, 0.0, 0.0])

    return X[:3] / w
