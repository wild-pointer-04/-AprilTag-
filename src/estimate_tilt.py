import cv2
import numpy as np
from src.utils import get_camera_intrinsics, rvec_to_euler_xyz, rvec_to_camera_tilt, normalize_angles, draw_axes


def build_obj_points(rows, cols, spacing=0.065, symmetric=True):
    if symmetric:
        xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
        obj = np.stack([xs, ys, np.zeros_like(xs)], axis=-1).reshape(-1,3)
        obj = obj * float(spacing)
    else:
        obj = []
        for i in range(rows):
            for j in range(cols):
                obj.append([ (2*j + i%2)*spacing, i*spacing, 0.0 ])
        obj = np.array(obj, dtype=np.float32)
    return obj.astype(np.float32)


def _generate_symmetry_variants(corners, rows, cols, enable_symmetry=True):
    """
    针对对称圆点阵，生成多种可能的行/列排列（旋转0/90/180/270°，以及每个旋转的左右镜像）。
    返回值: List[(corners_variant, description)]
    """
    corners = np.ascontiguousarray(corners, dtype=np.float32)
    base = corners.reshape(rows * cols, 2).reshape(rows, cols, 2)
    variants = []

    def _add_variant(arr, desc):
        variants.append(
            (
                np.ascontiguousarray(arr.reshape(-1, 1, 2), dtype=np.float32),
                desc
            )
        )

    if not enable_symmetry:
        _add_variant(base.copy(), 'identity')
        return variants

    if rows != cols:
        rotation_steps = [0, 2]  # 仅支持 0° 和 180°，避免非方阵形状不匹配
    else:
        rotation_steps = [0, 1, 2, 3]

    for k in rotation_steps:
        if k == 0:
            rotated = base.copy()
        else:
            rotated = np.rot90(base, k=k, axes=(0, 1))
        angle = (k * 90) % 360
        _add_variant(rotated.copy(), f'rot{angle}')
        _add_variant(np.flip(rotated, axis=1).copy(), f'rot{angle}_flipLR')

    return variants

def solve_pose_with_guess(gray, corners, rows, cols, spacing=10.0, symmetric=True, K=None, dist=None, camera_yaml_path=None):
    h, w = gray.shape[:2]
    if K is None or dist is None:
        # 优先从 YAML 文件加载真实内参，如果失败则使用默认值
        if camera_yaml_path is not None:
            K, dist = get_camera_intrinsics(h, w, yaml_path=camera_yaml_path, f_scale=1.0)
        else:
            K, dist = get_camera_intrinsics(h, w, f_scale=1.0)

    objp = build_obj_points(rows, cols, spacing, symmetric)
    assert objp.shape[0] == corners.shape[0], "2D/3D 点数不一致，请检查行列数。"

    corners = np.ascontiguousarray(corners, dtype=np.float32)

    if symmetric:
        candidates = _generate_symmetry_variants(corners, rows, cols, enable_symmetry=True)
    else:
        candidates = _generate_symmetry_variants(corners, rows, cols, enable_symmetry=False)

    best_solution = None
    objp_for_pnp = objp.reshape(-1, 3).astype(np.float32)

    for pts_variant, desc in candidates:
        try:
            ok, rvec_candidate, tvec_candidate = cv2.solvePnP(
                objp_for_pnp,
                pts_variant,
                K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except cv2.error:
            ok = False
        if not ok:
            continue

        proj, _ = cv2.projectPoints(objp_for_pnp, rvec_candidate, tvec_candidate, K, dist)
        residual = proj.reshape(-1, 2) - pts_variant.reshape(-1, 2)
        per_point = np.linalg.norm(residual, axis=1)
        mean_error = float(per_point.mean())

        if (best_solution is None) or (mean_error < best_solution['mean_error']):
            best_solution = {
                'rvec': rvec_candidate,
                'tvec': tvec_candidate,
                'mean_error': mean_error,
                'points': pts_variant,
                'transform': desc
            }

    if best_solution is None:
        raise RuntimeError("solvePnP 失败：所有对称排列均无法求解。")

    rvec = best_solution['rvec']
    tvec = best_solution['tvec']

    # 计算角度（提供两种方法）
    # 方法1: 标准欧拉角（板子相对于相机的旋转）
    roll_euler, pitch_euler, yaw_euler = rvec_to_euler_xyz(rvec)
    roll_euler, pitch_euler, yaw_euler = normalize_angles(roll_euler, pitch_euler, yaw_euler)
    
    # 方法2: 相机倾斜角（假设板子水平，计算相机相对于水平面的倾斜）
    roll_tilt, pitch_tilt, yaw_tilt = rvec_to_camera_tilt(rvec)
    roll_tilt, pitch_tilt, yaw_tilt = normalize_angles(roll_tilt, pitch_tilt, yaw_tilt)
    
    return rvec, tvec, {
        'euler': (roll_euler, pitch_euler, yaw_euler),  # 标准欧拉角
        'camera_tilt': (roll_tilt, pitch_tilt, yaw_tilt),  # 相机倾斜角（假设板子水平）
        'symmetry': {
            'transform': best_solution['transform'],
            'reprojection_error': best_solution['mean_error']
        }
    }, K, dist, best_solution['points']

def visualize_and_save(img_bgr, corners, K, dist, rvec, tvec, save_path, center_px=None, center_mean_px=None, blob_keypoints=None):
    vis = img_bgr.copy()
    
    # 先绘制所有 blob 检测到的点（绿色，较大）
    if blob_keypoints is not None:
        print(f"[DEBUG] 可视化: 绘制 {len(blob_keypoints)} 个 blob 检测点（绿色）")
        for kp in blob_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = float(kp.size) if hasattr(kp, 'size') else 5.0
            radius = max(2, int(round(size / 2.0)))
            cv2.circle(vis, (x, y), radius, (0, 255, 0), 2)
    
    # 然后绘制网格匹配成功的点（黄色，较小，会覆盖在绿色点上）
    for p in corners.reshape(-1,2):
        cv2.circle(vis, tuple(np.round(p).astype(int)), 3, (0,255,255), -1)
    # 可选：绘制板子中心
    if center_px is not None:
        c = tuple(np.round(center_px).astype(int))
        cv2.drawMarker(vis, c, (255, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=24, thickness=2)
    if center_mean_px is not None:
        cm = tuple(np.round(center_mean_px).astype(int))
        cv2.drawMarker(vis, cm, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
    vis = draw_axes(vis, K, dist, rvec, tvec, axis_len=150)
    # 打印坐标轴端点在图像中的像素坐标，以及在相机坐标系下的深度Z（以板子原点为 [0,0,0]）
    axis_len = 150
    axis = np.float32([
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, -axis_len],
        [0, 0, 0]
    ]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    R, _ = cv2.Rodrigues(rvec)
    cam_pts = (R @ axis.T + tvec.reshape(3,1)).T
    names = ["X_end", "Y_end", "Z_end", "Origin"]
    print("\n== 坐标轴端点像素坐标与相机系深度 ==")
    for i, name in enumerate(names):
        u, v = imgpts[i]
        Xc, Yc, Zc = cam_pts[i]
        print(f"{name:6s}: pixel=({u:.1f}, {v:.1f}), camera XYZ=({Xc:.1f}, {Yc:.1f}, {Zc:.1f})")
    cv2.imwrite(save_path, vis)
    return save_path
