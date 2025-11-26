import cv2
import numpy as np
from src.utils import build_blob_detector


def try_find(gray, rows, cols, symmetric=True):
    flags = cv2.CALIB_CB_CLUSTERING
    flags |= (cv2.CALIB_CB_SYMMETRIC_GRID if symmetric else cv2.CALIB_CB_ASYMMETRIC_GRID)
    det = build_blob_detector()
    keypoints = det.detect(gray)  # 检测 blob 点
    ok, centers = cv2.findCirclesGrid(gray, (cols, rows), flags=flags, blobDetector=det)
    return ok, centers, keypoints  # 返回 blob 点用于可视化

def refine(gray, pts):
    g = cv2.GaussianBlur(gray, (5,5), 0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    return cv2.cornerSubPix(g, pts.astype(np.float32), (5,5), (-1,-1), crit)

def auto_search(gray, rows_range=(8, 28), cols_range=(8, 28)):
    """
    自动枚举 rows/cols 与对称性。返回第一个命中的组合及坐标。
    你也可以固定 rows/cols 以提速。
    """
    det = build_blob_detector()
    keypoints = det.detect(gray)  # 检测 blob 点用于可视化
    for symmetric in (True, False):
        for r in range(rows_range[0], rows_range[1]+1):
            for c in range(cols_range[0], cols_range[1]+1):
                ok, centers, _ = try_find(gray, r, c, symmetric=symmetric)
                if ok:
                    return True, refine(gray, centers), (r, c, symmetric), keypoints
    return False, None, None, keypoints
