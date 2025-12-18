#!/usr/bin/env python3
"""
AprilTag 引导的相机标定 - 完美融合版
结合Version4的单应性补全 + Version8的Ref点原点坐标系
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from scipy.spatial import cKDTree

try:
    from pupil_apriltags import Detector
    USING_PUPIL = True
except ImportError:
    import apriltag
    USING_PUPIL = False


def load_camera_params(yaml_path):
    """加载相机内参"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        dist_coeffs = np.array(data['distortion_coefficients']['data'])
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"❌ 加载相机参数失败: {e}")
        return None, None


def detect_apriltag(gray_image):
    """检测AprilTag"""
    if USING_PUPIL:
        detector = Detector(
            families='tagStandard41h12',
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True
        )
    else:
        options = apriltag.DetectorOptions(families='tagStandard41h12')
        detector = apriltag.Detector(options)

    detections = detector.detect(gray_image)
    if len(detections) == 0:
        return None

    detection = detections[0]
    corners = np.array(detection.corners, dtype=np.float64)
    center = np.array(detection.center, dtype=np.float64)
    tag_id = detection.tag_id
    return {'corners': corners, 'center': center, 'tag_id': tag_id}


def detect_blobs(gray_image):
    """检测圆形blob"""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 80
    params.maxArea = 3000
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = True
    params.minConvexity = 0.6
    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_image)
    blob_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    print(f"\n[Blob检测] 共检测到 {len(blob_points)} 个候选点")
    return blob_points, keypoints


def filter_blobs_exclude_apriltag(blob_points, apriltag_info):
    """排除AprilTag区域"""
    if len(blob_points) == 0:
        return blob_points, np.array([])
    
    tag_center = apriltag_info['center']
    tag_corners = apriltag_info['corners']
    tag_size_px = np.mean([
        np.linalg.norm(tag_corners[1] - tag_corners[0]),
        np.linalg.norm(tag_corners[2] - tag_corners[1])
    ])
    
    exclusion_radius = tag_size_px * 2.0
    distances = np.linalg.norm(blob_points - tag_center, axis=1)
    valid_mask = distances > exclusion_radius
    filtered_blobs = blob_points[valid_mask]
    excluded_indices = np.where(~valid_mask)[0]
    return filtered_blobs, excluded_indices

def find_all_virtual_corners(blob_points, width_height_ratio=1.0):
    """
    【全局拟合】同时拟合四条边，计算四个虚拟角点
    这是解决大角度透视、部分遮挡的终极方案。
    """
    # 1. 使用 minAreaRect 获取粗略的边缘归属
    rect = cv2.minAreaRect(blob_points)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")
    
    # 排序：左上、右上、右下、左下
    # (这里用简单的 Y 轴排序 + X 轴排序)
    sorted_y = box[np.argsort(box[:, 1])]
    top_2 = sorted_y[:2]
    bottom_2 = sorted_y[2:]
    
    # 细分左右
    tl_ideal = top_2[np.argmin(top_2[:, 0])]
    tr_ideal = top_2[np.argmax(top_2[:, 0])]
    bl_ideal = bottom_2[np.argmin(bottom_2[:, 0])]
    br_ideal = bottom_2[np.argmax(bottom_2[:, 0])]
    
    ideal_corners = [tl_ideal, tr_ideal, br_ideal, bl_ideal]
    
    # 定义四条边的端点对 (索引)
    # 0:Top(TL-TR), 1:Right(TR-BR), 2:Bottom(BR-BL), 3:Left(BL-TL)
    edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edge_names = ["Top", "Right", "Bottom", "Left"]
    
    lines = [] # 存储拟合出的直线 [vx, vy, x, y]
    
    print("\n[全局拟合] 正在拟合四条边界...")
    
    for i, (idx1, idx2) in enumerate(edge_indices):
        p1 = ideal_corners[idx1]
        p2 = ideal_corners[idx2]
        
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        
        if edge_len < 10: # 边太短，异常
            lines.append(None)
            continue

        # 收集属于这条边的点
        edge_points = []
        for pt in blob_points:
            vec_p1_pt = pt - p1
            proj_len = np.dot(vec_p1_pt, edge_vec) / edge_len
            dist = np.linalg.norm(vec_p1_pt - (edge_vec * proj_len / edge_len))
            
            # 阈值：距离直线 < 20px，投影在[-10%, 110%]范围内
            if dist < 20 and -0.1 * edge_len < proj_len < 1.1 * edge_len:
                edge_points.append(pt)
        
        edge_points = np.array(edge_points)
        
        if len(edge_points) < 5:
            print(f"  ⚠️ {edge_names[i]} 边点数过少 ({len(edge_points)})，使用粗略边")
            vx, vy = (p2 - p1) / edge_len
            x, y = p1
        else:
            [vx, vy, x, y] = cv2.fitLine(edge_points, cv2.DIST_L2, 0, 0.01, 0.01)
            
        lines.append((vx, vy, x, y))
        
    # 计算四条直线的交点
    # Top x Left -> TL
    # Top x Right -> TR
    # Bottom x Right -> BR
    # Bottom x Left -> BL
    
    def get_intersection(line1, line2):
        if line1 is None or line2 is None: return None
        (vx1, vy1, x1, y1) = line1
        (vx2, vy2, x2, y2) = line2
        
        A1, B1, C1 = -vy1, vx1, -vy1*x1 + vx1*y1
        A2, B2, C2 = -vy2, vx2, -vy2*x2 + vx2*y2
        
        det = A1*B2 - A2*B1
        if abs(det) < 1e-6: return None
        
        ix = (B2*C1 - B1*C2) / det
        iy = (A1*C2 - A2*C1) / det
        return np.array([ix, iy], dtype=np.float32).flatten()

    final_corners = {}
    final_corners['top_left'] = get_intersection(lines[0], lines[3])
    final_corners['top_right'] = get_intersection(lines[0], lines[1])
    final_corners['bottom_right'] = get_intersection(lines[2], lines[1])
    final_corners['bottom_left'] = get_intersection(lines[2], lines[3])
    
    # 检查是否有 None
    for k, v in final_corners.items():
        if v is None:
            print(f"  ❌ 无法计算交点: {k}")
            return None
            
    print(f"  ✅ 四角拟合成功！")
    return final_corners


def build_grid_by_homography(virtual_corners, pattern_size=(15, 15)):
    """
    【透视映射】
    利用拟合出的四个角点，通过 Homography 一次性生成所有网格点。
    这是对抗透视畸变的终极方法。
    """
    cols, rows = pattern_size
    
    # 1. 定义完美的物理网格坐标 (归一化)
    # 假设左上角是(0,0)，右下角是(cols-1, rows-1)
    src_pts = np.float32([
        [0, 0],             # TL
        [cols-1, 0],        # TR
        [cols-1, rows-1],   # BR
        [0, rows-1]         # BL
    ]).reshape(-1, 1, 2)
    
    # 2. 提取拟合出的图像角点
    dst_pts = np.float32([
        virtual_corners['top_left'],
        virtual_corners['top_right'],
        virtual_corners['bottom_right'],
        virtual_corners['bottom_left']
    ]).reshape(-1, 1, 2)
    
    # 3. 计算单应性矩阵
    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    # 4. 生成所有 225 个点的理想坐标
    grid_indices = []
    for r in range(rows):
        for c in range(cols):
            grid_indices.append([c, r]) # x=col, y=row
            
    grid_indices = np.array(grid_indices, dtype=np.float32).reshape(-1, 1, 2)
    
    # 5. 透视变换！
    # 这一步会自动把 "近大远小" 的效果算进去
    projected_points = cv2.perspectiveTransform(grid_indices, H)
    
    # 6. 整理形状
    grid_points = projected_points.reshape(rows, cols, 2)
    
    return grid_points

def filter_blobs_by_grid_pattern(blob_points, apriltag_info):
    """基于网格模式过滤"""
    if len(blob_points) < 100:
        return blob_points, np.array([])
    
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(blob_points)
        distances, _ = tree.query(blob_points, k=5)
        nearest_distances = distances[:, 1]
        estimated_spacing = np.median(nearest_distances)
        
        min_spacing = estimated_spacing * 0.3
        max_spacing = estimated_spacing * 1.4
        
        valid_neighbor_counts = []
        for i, point in enumerate(blob_points):
            neighbors = tree.query_ball_point(point, r=max_spacing)
            valid_neighbors = 0
            for j in neighbors:
                if i == j:
                    continue
                dist = np.linalg.norm(blob_points[i] - blob_points[j])
                if min_spacing <= dist <= max_spacing:
                    valid_neighbors += 1
            valid_neighbor_counts.append(valid_neighbors)
        
        valid_neighbor_counts = np.array(valid_neighbor_counts)
        valid_mask = valid_neighbor_counts >= 1
        filtered_blobs = blob_points[valid_mask]
        outlier_indices = np.where(~valid_mask)[0]
        return filtered_blobs, outlier_indices
    except ImportError:
        return blob_points, np.array([])


def find_four_corner_blobs(blob_points):
    """找到标定板的四个角点"""
    if len(blob_points) < 4:
        return None, None

    rect = cv2.minAreaRect(blob_points)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")

    sorted_y_indices = np.argsort(box[:, 1])
    top_2_indices = sorted_y_indices[:2]
    bottom_2_indices = sorted_y_indices[2:]
    
    top_2 = box[top_2_indices]
    bottom_2 = box[bottom_2_indices]
    
    if top_2[0][0] < top_2[1][0]:
        tl_ideal, tr_ideal = top_2[0], top_2[1]
    else:
        tl_ideal, tr_ideal = top_2[1], top_2[0]
        
    if bottom_2[0][0] < bottom_2[1][0]:
        bl_ideal, br_ideal = bottom_2[0], bottom_2[1]
    else:
        bl_ideal, br_ideal = bottom_2[1], bottom_2[0]

    corner_positions = {
        'top_left': tl_ideal,
        'top_right': tr_ideal,
        'bottom_left': bl_ideal,
        'bottom_right': br_ideal
    }
    
    corners_dict = {}
    corners_indices = {}
    used_indices = set()
    
    print(f"\n[查找四个角点]")
    for corner_name, ideal_pos in corner_positions.items():
        distances = np.linalg.norm(blob_points - ideal_pos, axis=1)
        sorted_dist_indices = np.argsort(distances)
        
        found_idx = -1
        for idx in sorted_dist_indices:
            if idx not in used_indices:
                found_idx = idx
                break
        
        if found_idx == -1:
            found_idx = sorted_dist_indices[0]
            
        nearest_blob = blob_points[found_idx]
        used_indices.add(found_idx)
        corners_dict[corner_name] = nearest_blob
        corners_indices[corner_name] = found_idx
        print(f"  {corner_name}: ({nearest_blob[0]:.1f}, {nearest_blob[1]:.1f})")
    
    return corners_dict, corners_indices


def find_corner_near_apriltag(corners_dict, corners_indices, apriltag_info):
    """
    找到最靠近AprilTag的角点作为Ref（保持向后兼容）
    如果 AprilTag 固定在右上，则优先返回 top_right
    """
    # 固定场景：AprilTag在右上角，直接返回 top_right，更稳
    if 'top_right' in corners_dict:
        return corners_dict['top_right'], corners_indices['top_right'], 'top_right'
    
    tag_center = apriltag_info['center']
    min_dist = float('inf')
    nearest_corner_name = None
    nearest_corner_blob = None
    nearest_corner_idx = None
    
    for corner_name, corner_blob in corners_dict.items():
        dist = np.linalg.norm(corner_blob - tag_center)
        if dist < min_dist:
            min_dist = dist
            nearest_corner_name = corner_name
            nearest_corner_blob = corner_blob
            nearest_corner_idx = corners_indices[corner_name]
    
    return nearest_corner_blob, nearest_corner_idx, nearest_corner_name


def get_grid_orientation_from_line_fitting(blob_points):
    """
    【鲁棒方法】使用木匠拉线法计算网格方向
    通过拟合多条边缘直线来计算，不依赖四个角点
    """
    # 1. 先用 minAreaRect 得到一个粗略的矩形方向
    rect = cv2.minAreaRect(blob_points)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")
    
    # 对矩形的四个角排序
    sorted_y = box[np.argsort(box[:, 1])]
    top_2 = sorted_y[:2]
    bottom_2 = sorted_y[2:]
    
    if top_2[0][0] < top_2[1][0]:
        tl, tr = top_2[0], top_2[1]
    else:
        tl, tr = top_2[1], top_2[0]
        
    if bottom_2[0][0] < bottom_2[1][0]:
        bl, br = bottom_2[0], bottom_2[1]
    else:
        bl, br = bottom_2[1], bottom_2[0]
    
    # 2. 拟合四条边的直线
    edges = [
        (tl, tr, "top"),      # 上边
        (bl, br, "bottom"),   # 下边
        (tl, bl, "left"),     # 左边
        (tr, br, "right")     # 右边
    ]
    
    horizontal_vecs = []
    vertical_vecs = []
    
    print(f"\n[方向计算-木匠拉线法] 拟合边缘直线计算网格方向")
    
    for p1, p2, edge_name in edges:
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0:
            continue
        
        # 找到属于这条边的点
        edge_points = []
        for pt in blob_points:
            vec_p1_pt = pt - p1
            proj_len = np.dot(vec_p1_pt, edge_vec) / edge_len
            dist = np.linalg.norm(vec_p1_pt - (edge_vec * proj_len / edge_len))
            
            if dist < 15 and -edge_len*0.1 < proj_len < edge_len*1.1:
                edge_points.append(pt)
        
        if len(edge_points) < 5:
            # 点太少，用粗略的边
            direction = edge_vec / edge_len
        else:
            # 拟合直线
            edge_points = np.array(edge_points)
            [vx, vy, x, y] = cv2.fitLine(edge_points, cv2.DIST_L2, 0, 0.01, 0.01)
            direction = np.array([vx[0], vy[0]])
        
        # 分类到水平或垂直
        if edge_name in ["top", "bottom"]:
            horizontal_vecs.append(direction)
        else:
            vertical_vecs.append(direction)
    
    # 3. 合并同方向的向量
    if len(horizontal_vecs) == 0 or len(vertical_vecs) == 0:
        # 兜底：使用粗略矩形
        unit_x = (tr - tl) / np.linalg.norm(tr - tl)
        unit_y = (bl - tl) / np.linalg.norm(bl - tl)
        print(f"  ⚠️ 拟合点不足，使用粗略矩形方向")
    else:
        # 对水平向量取平均（注意方向一致性）
        h_vec = horizontal_vecs[0]
        for v in horizontal_vecs[1:]:
            if np.dot(h_vec, v) < 0:  # 方向相反，翻转
                v = -v
            h_vec = h_vec + v
        unit_x = h_vec / np.linalg.norm(h_vec)
        
        # 对垂直向量取平均
        v_vec = vertical_vecs[0]
        for v in vertical_vecs[1:]:
            if np.dot(v_vec, v) < 0:
                v = -v
            v_vec = v_vec + v
        unit_y = v_vec / np.linalg.norm(v_vec)
        
        print(f"  ✅ 拟合成功 (使用{len(horizontal_vecs)}条水平边, {len(vertical_vecs)}条垂直边)")
    
    print(f"  X方向(水平): {unit_x}")
    print(f"  Y方向(垂直): {unit_y}")
    
    return unit_x, unit_y


def find_virtual_corner_by_lines(blob_points, corner_name='top_right'):
    """
    【方案三：木匠拉线法】
    通过拟合边缘直线来计算虚拟角点（即使物理角点不存在也能算出位置）
    """
    # 1. 先用 minAreaRect 得到一个粗略的矩形，作为参考系
    rect = cv2.minAreaRect(blob_points)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")
    
    # 对矩形的四个角排序
    sorted_y = box[np.argsort(box[:, 1])]
    top_2 = sorted_y[:2]
    bottom_2 = sorted_y[2:]
    
    # 再次按X排序分左右
    if top_2[0][0] < top_2[1][0]:
        tl_ideal, tr_ideal = top_2[0], top_2[1]
    else:
        tl_ideal, tr_ideal = top_2[1], top_2[0]
        
    if bottom_2[0][0] < bottom_2[1][0]:
        bl_ideal, br_ideal = bottom_2[0], bottom_2[1]
    else:
        bl_ideal, br_ideal = bottom_2[1], bottom_2[0]
    
    ideal_corners = {'top_left': tl_ideal, 'top_right': tr_ideal, 
                     'bottom_left': bl_ideal, 'bottom_right': br_ideal}
    
    # 2. 确定我们要拟合哪两条边
    target_edges = []
    
    if corner_name == 'top_right':
        target_edges = [
            (ideal_corners['top_left'], ideal_corners['top_right']),   # 上边
            (ideal_corners['bottom_right'], ideal_corners['top_right']) # 右边
        ]
    elif corner_name == 'top_left':
        target_edges = [
            (ideal_corners['top_right'], ideal_corners['top_left']),   # 上边
            (ideal_corners['bottom_left'], ideal_corners['top_left'])  # 左边
        ]
    elif corner_name == 'bottom_left':
        target_edges = [
            (ideal_corners['top_left'], ideal_corners['bottom_left']), # 左边
            (ideal_corners['bottom_right'], ideal_corners['bottom_left']) # 下边
        ]
    elif corner_name == 'bottom_right':
        target_edges = [
            (ideal_corners['bottom_left'], ideal_corners['bottom_right']), # 下边
            (ideal_corners['top_right'], ideal_corners['bottom_right'])    # 右边
        ]
    
    lines_params = []
    print(f"\n[木匠拉线] 正在拟合直线以寻找虚拟 {corner_name}...")
    
    # 3. 循环拟合两条边
    for p1, p2 in target_edges:
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0: continue
        
        # 挑选属于这条边的点
        edge_points = []
        for pt in blob_points:
            vec_p1_pt = pt - p1
            proj_len = np.dot(vec_p1_pt, edge_vec) / edge_len
            dist = np.linalg.norm(vec_p1_pt - (edge_vec * proj_len / edge_len))
            
            if dist < 15 and -edge_len*0.1 < proj_len < edge_len*1.1:
                edge_points.append(pt)
        
        edge_points = np.array(edge_points)
        
        if len(edge_points) < 5:
            print(f"  ⚠️ 警告：某条边只找到了 {len(edge_points)} 个点，拟合可能不准！")
            vx, vy = (p2 - p1) / np.linalg.norm(p2 - p1)
            x, y = p2
        else:
            [vx, vy, x, y] = cv2.fitLine(edge_points, cv2.DIST_L2, 0, 0.01, 0.01)
            
        lines_params.append((vx, vy, x, y))

    if len(lines_params) < 2:
        return None

    # 4. 计算两条直线的交点
    (vx1, vy1, x1, y1) = lines_params[0]
    (vx2, vy2, x2, y2) = lines_params[1]
    
    A1, B1 = -vy1, vx1
    C1 = A1*x1 + B1*y1
    
    A2, B2 = -vy2, vx2
    C2 = A2*x2 + B2*y2
    
    det = A1*B2 - A2*B1
    if abs(det) < 1e-6:
        print("  ❌ 错误：两条直线平行，无法计算交点！")
        return None
        
    intersect_x = (B2*C1 - B1*C2) / det
    intersect_y = (A1*C2 - A2*C1) / det
    
    virtual_point = np.array([intersect_x, intersect_y], dtype=np.float32).reshape(2)
    print(f"  ✅ 拟合成功！虚拟交点坐标: ({intersect_x[0]:.2f}, {intersect_y[0]:.2f})")
    
    return virtual_point


def estimate_spacing_robust(blob_points, pattern_size=(15, 15)):
    """
    【鲁棒方法】从所有检测到的blob点估算圆点间距
    使用KNN近邻距离统计，不依赖四个角点
    """
    if len(blob_points) < 10:
        return None
    
    try:
        from scipy.spatial import cKDTree
        
        # 1. 建立KD树查找每个点的最近邻
        tree = cKDTree(blob_points)
        
        # 查找每个点最近的5个邻居（k=6因为包含自己）
        distances, indices = tree.query(blob_points, k=min(6, len(blob_points)))
        
        # 2. 提取最近邻距离（排除自己，即distances[:, 0]）
        nearest_distances = distances[:, 1]  # 第1近的邻居
        
        # 3. 使用中位数估算间距（比均值更鲁棒，不受离群点影响）
        spacing_median = np.median(nearest_distances)
        
        # 4. 为了更准确，我们也计算第2、3近的邻居距离
        if distances.shape[1] >= 3:
            second_nearest = distances[:, 2]
            # 过滤掉明显的对角线距离（约为spacing * sqrt(2)）
            # 只保留接近spacing的距离
            valid_second = second_nearest[second_nearest < spacing_median * 1.4]
            if len(valid_second) > 0:
                spacing_from_second = np.median(valid_second)
                # 加权平均
                spacing_px = (spacing_median * 0.7 + spacing_from_second * 0.3)
            else:
                spacing_px = spacing_median
        else:
            spacing_px = spacing_median
        
        # 5. 验证合理性
        spacing_std = np.std(nearest_distances)
        
        print(f"\n[间距估算-鲁棒方法] 基于{len(blob_points)}个点的近邻分析")
        print(f"  最近邻距离: 中位数={spacing_median:.2f}px, 标准差={spacing_std:.2f}px")
        print(f"  最终估算间距: {spacing_px:.2f}px")
        
        # 合理性检查
        if spacing_px < 10 or spacing_px > 500:
            print(f"  ⚠️ 警告：估算间距异常 ({spacing_px:.2f}px)")
            return None
        
        return spacing_px
        
    except ImportError:
        # 如果没有scipy，使用简单方法
        print(f"\n[间距估算-简单方法] scipy不可用，使用暴力计算")
        
        # 计算所有点对之间的距离（仅采样部分避免过慢）
        sample_size = min(100, len(blob_points))
        sample_indices = np.random.choice(len(blob_points), sample_size, replace=False)
        sample_points = blob_points[sample_indices]
        
        min_distances = []
        for pt in sample_points:
            dists = np.linalg.norm(blob_points - pt, axis=1)
            dists_sorted = np.sort(dists)
            if len(dists_sorted) > 1:
                min_distances.append(dists_sorted[1])  # 排除自己（距离0）
        
        spacing_px = np.median(min_distances)
        print(f"  采样{sample_size}个点, 估算间距: {spacing_px:.2f}px")
        
        return spacing_px


def build_grid_by_walking(ref_blob, corner_name, unit_x, unit_y,
                          spacing_px, filtered_blobs, pattern_size=(15, 15)):
    """
    【爬虫策略】顺藤摸瓜构建网格
    利用 KD-Tree，从 Ref 点开始，一个接一个地寻找真实的相邻点。
    不再依赖全局线性公式，完美解决透视和畸变问题。
    """
    rows, cols = pattern_size
    
    # 1. 构建 KD-Tree 用于快速查找最近邻
    from scipy.spatial import cKDTree
    tree = cKDTree(filtered_blobs)
    
    # 初始化网格矩阵
    grid_points = np.zeros((rows, cols, 2), dtype=np.float32)
    # 状态矩阵：1表示找到了真实点，0表示靠推算的
    status_mask = np.zeros((rows, cols), dtype=np.int8)
    
    # 2. 确定 Ref 点的逻辑坐标 (r, c)
    ref_positions = {
        'top_left': (0, 0),
        'top_right': (0, cols - 1),
        'bottom_left': (rows - 1, 0),
        'bottom_right': (rows - 1, cols - 1)
    }
    start_r, start_c = ref_positions.get(corner_name, (0, cols - 1))
    
    # 先把 Ref 点填进去
    grid_points[start_r, start_c] = ref_blob
    status_mask[start_r, start_c] = 1
    
    # 定义搜索半径：步长的一半 (太大了会串行，太小了找不到)
    search_radius = spacing_px * 0.6
    
    print(f"\n[网格爬虫] 从 ({start_r},{start_c}) 开始顺藤摸瓜...")

    # =========================================================
    # 策略：先竖向爬行（构建主干），再横向爬行（构建枝叶）
    # =========================================================
    
    # 3. 确定竖向爬行的方向 (向上还是向下)
    # 如果 Ref 在上面(row=0)，就向下爬(step=1)；如果在下面，就向上爬(step=-1)
    row_step = 1 if start_r == 0 else -1
    col_step = 1 if start_c == 0 else -1 # 横向同理
    
    # --- 第一阶段：沿着 Ref 所在的这一列，把“脊椎”建立起来 ---
    # 比如 Ref 是右上角，我们先沿着最右边这一列，从上往下找
    current_r = start_r
    
    # 循环填满这一列
    while 0 <= current_r + row_step < rows:
        prev_blob = grid_points[current_r, start_c]
        next_r = current_r + row_step
        
        # 预测下一个点的位置 (先用理论方向猜一下)
        predicted_point = prev_blob + unit_y * row_step * spacing_px
        
        # 在预测点附近找有没有真的斑点
        # k=1 找最近的一个
        dist, idx = tree.query(predicted_point, k=1)
        
        if dist < search_radius:
            # 找到了！用真实的斑点
            found_blob = filtered_blobs[idx]
            grid_points[next_r, start_c] = found_blob
            status_mask[next_r, start_c] = 1
            
            # 【关键】动态更新方向！
            # 用新找到的点和上一个点的连线，作为下一步的方向
            # 这样就能顺着透视的曲线走了
            actual_vec = found_blob - prev_blob
            unit_y = actual_vec / np.linalg.norm(actual_vec) # 更新 unit_y
        else:
            # 没找到（可能被遮挡），只能用预测点凑合
            grid_points[next_r, start_c] = predicted_point
            status_mask[next_r, start_c] = 0 # 标记为虚拟点
            
        current_r = next_r

    # --- 第二阶段：以这一列为基础，逐行横向爬行 ---
    # 现在最右边那一列都有坐标了，我们对每一行，从右向左爬
    
    for r in range(rows):
        # 这一行的起点是刚才建立的“脊椎”
        current_c = start_c
        
        # 重置横向方向 (每一行开始都用初始方向，避免上一行的误差带过来)
        # 注意：这里可以用 neighbors 优化，但简化起见用全局 unit_x 修正
        curr_unit_x = unit_x 
        
        while 0 <= current_c + col_step < cols:
            prev_blob = grid_points[r, current_c]
            next_c = current_c + col_step
            
            # 预测
            predicted_point = prev_blob + curr_unit_x * col_step * spacing_px
            
            # 搜索
            dist, idx = tree.query(predicted_point, k=1)
            
            if dist < search_radius:
                # 找到了
                found_blob = filtered_blobs[idx]
                grid_points[r, next_c] = found_blob
                status_mask[r, next_c] = 1
                
                # 动态更新横向方向
                actual_vec = found_blob - prev_blob
                curr_unit_x = actual_vec / np.linalg.norm(actual_vec)
            else:
                # 没找到
                grid_points[r, next_c] = predicted_point
                status_mask[r, next_c] = 0
            
            current_c = next_c

    return grid_points



def estimate_spacing_from_corners(corners_dict, pattern_size=(15, 15)):
    """
    从四个角点估算圆点间距
    更准确，因为考虑了整个标定板的尺寸
    """
    cols, rows = pattern_size
    
    tl = corners_dict['top_left']
    tr = corners_dict['top_right']
    bl = corners_dict['bottom_left']
    br = corners_dict['bottom_right']
    
    # 水平方向：top_left 到 top_right 的距离 / (cols-1)
    horizontal_dist_top = np.linalg.norm(tr - tl)
    horizontal_dist_bottom = np.linalg.norm(br - bl)
    spacing_horizontal = (horizontal_dist_top + horizontal_dist_bottom) / 2 / (cols - 1)
    
    # 垂直方向：top_left 到 bottom_left 的距离 / (rows-1)
    vertical_dist_left = np.linalg.norm(bl - tl)
    vertical_dist_right = np.linalg.norm(br - tr)
    spacing_vertical = (vertical_dist_left + vertical_dist_right) / 2 / (rows - 1)
    
    # 取平均
    spacing_px = (spacing_horizontal + spacing_vertical) / 2
    
    print(f"\n[间距估算] 从四角点计算")
    print(f"  水平间距: {spacing_horizontal:.2f}px")
    print(f"  垂直间距: {spacing_vertical:.2f}px")
    print(f"  平均间距: {spacing_px:.2f}px")
    
    return spacing_px

def simple_greedy_matching(cost_matrix, threshold=1e6):
    """贪心匹配"""
    n_rows, n_cols = cost_matrix.shape
    matches = []
    
    for i in range(n_rows):
        for j in range(n_cols):
            if cost_matrix[i, j] < threshold:
                matches.append((cost_matrix[i, j], i, j))
    
    matches.sort()
    used_rows, used_cols = set(), set()
    result_rows, result_cols = [], []
    
    for cost, i, j in matches:
        if i not in used_rows and j not in used_cols:
            result_rows.append(i)
            result_cols.append(j)
            used_rows.add(i)
            used_cols.add(j)
    
    return np.array(result_rows), np.array(result_cols)


def match_blobs_to_grid(blob_points, grid_points, max_distance=200.0):
    """匹配blob到网格（简单贪心，确保Ref列行保持）"""
    rows, cols = grid_points.shape[:2]
    
    valid_grids = []
    valid_indices = []
    for row in range(rows):
        for col in range(cols):
            valid_grids.append(grid_points[row, col])
            valid_indices.append((row, col))
    
    if len(valid_grids) == 0 or len(blob_points) == 0:
        return None, None
    
    valid_grids = np.array(valid_grids)
    valid_indices = np.array(valid_indices)
    
    n_grids = len(valid_grids)
    n_blobs = len(blob_points)
    cost_matrix = np.zeros((n_grids, n_blobs), dtype=np.float32)
    
    for i in range(n_grids):
        for j in range(n_blobs):
            dist = np.linalg.norm(valid_grids[i] - blob_points[j])
            cost_matrix[i, j] = dist if dist <= max_distance else 1e6
    
    grid_indices, blob_indices = simple_greedy_matching(cost_matrix, threshold=max_distance)
    
    matched_corners = []
    matched_indices = []
    
    for grid_idx, blob_idx in zip(grid_indices, blob_indices):
        if cost_matrix[grid_idx, blob_idx] < max_distance:
            row, col = valid_indices[grid_idx]
            matched_corners.append(blob_points[blob_idx])
            matched_indices.append([row, col])
    
    if len(matched_corners) == 0:
        return None, None
    
    matched_corners = np.array(matched_corners, dtype=np.float32).reshape(-1, 1, 2)
    matched_indices = np.array(matched_indices, dtype=np.int32)
    
    return matched_corners, matched_indices


def refine_grid_with_homography(grid_points, matched_indices, matched_corners,
                               pattern_size=(15, 15), ref_grid_pos=None, ref_point=None):
    """
    使用单应性矩阵优化网格位置
    这是Version 4的核心 - 保证补全点位置准确
    """
    if len(matched_corners) < 4:
        return grid_points, None
    
    src_points = []
    dst_points = []
    
    for i in range(len(matched_indices)):
        row, col = matched_indices[i]
        src_points.append([col, row])
        dst_points.append(matched_corners[i, 0])
    
    # 强制加入Ref约束，确保单应性不会偏移原点
    if ref_grid_pos is not None and ref_point is not None:
        ref_r, ref_c = ref_grid_pos
        src_points.append([ref_c, ref_r])
        dst_points.append(ref_point)
    
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    if H is None:
        return grid_points, None
    
    rows, cols = pattern_size
    all_grid_logical = []
    for r in range(rows):
        for c in range(cols):
            all_grid_logical.append([c, r])
    
    all_grid_logical = np.array(all_grid_logical, dtype=np.float32).reshape(-1, 1, 2)
    refined_points_flat = cv2.perspectiveTransform(all_grid_logical, H)
    refined_grid_points = refined_points_flat.reshape(rows, cols, 2)
    
    print(f"[单应性优化] 使用{len(matched_corners)}个匹配点计算单应性矩阵")
    
    return refined_grid_points, H


def match_and_fill_missing_points(refined_grid_points, all_blobs, 
                                  search_radius=20.0, pattern_size=(15, 15)):
    """
    智能补全网格 - Version 4的方法
    返回最终匹配的blob点及其网格索引
    """
    rows, cols = pattern_size
    final_grid = np.zeros_like(refined_grid_points)
    status_mask = np.zeros((rows, cols), dtype=np.uint8)
    used_blobs = set()
    
    matched_count = 0
    interpolated_count = 0
    
    # 记录最终匹配的点
    final_matched_blobs = []
    final_matched_grid_indices = []
    
    for r in range(rows):
        for c in range(cols):
            theoretical_pt = refined_grid_points[r, c]
            
            if len(all_blobs) > 0:
                dists = np.linalg.norm(all_blobs - theoretical_pt, axis=1)
                min_dist_idx = np.argmin(dists)
                min_dist = dists[min_dist_idx]
                
                if min_dist < search_radius and min_dist_idx not in used_blobs:
                    final_grid[r, c] = all_blobs[min_dist_idx]
                    status_mask[r, c] = 1
                    used_blobs.add(min_dist_idx)
                    matched_count += 1
                    # 记录匹配信息
                    final_matched_blobs.append(all_blobs[min_dist_idx])
                    final_matched_grid_indices.append([r, c])
                else:
                    final_grid[r, c] = theoretical_pt
                    status_mask[r, c] = 0
                    interpolated_count += 1
            else:
                final_grid[r, c] = theoretical_pt
                status_mask[r, c] = 0
                interpolated_count += 1
    
    print(f"[智能补全] 匹配: {matched_count}, 插值: {interpolated_count}")
    
    # 转换为numpy数组
    final_matched_blobs = np.array(final_matched_blobs, dtype=np.float32) if final_matched_blobs else np.array([])
    final_matched_grid_indices = np.array(final_matched_grid_indices, dtype=np.int32) if final_matched_grid_indices else np.array([])
    
    return final_grid, status_mask, matched_count, interpolated_count, final_matched_blobs, final_matched_grid_indices


def get_ref_grid_position(corner_name, pattern_size=(15, 15)):
    """获取Ref点在标准网格中的位置"""
    cols, rows = pattern_size
    
    ref_positions = {
        'top_left': (0, 0),
        'top_right': (0, cols - 1),
        'bottom_left': (rows - 1, 0),
        'bottom_right': (rows - 1, cols - 1)
    }
    
    return ref_positions.get(corner_name, (0, cols - 1))


def solve_pnp_with_ref_origin(final_grid_points, ref_grid_pos, camera_matrix,
                               dist_coeffs, circle_spacing_m=0.02, pattern_size=(15, 15)):
    """
    PnP求解 - 以Ref点为原点 (Version 8的方法)
    """
    rows, cols = pattern_size
    ref_row, ref_col = ref_grid_pos
    
    object_points = []
    image_points = []
    
    for r in range(rows):
        for c in range(cols):
            # 3D坐标相对于Ref点
            x_3d = (c - ref_col) * circle_spacing_m
            y_3d = (r - ref_row) * circle_spacing_m
            z_3d = 0.0
            
            object_points.append([x_3d, y_3d, z_3d])
            image_points.append(final_grid_points[r, c])
    
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    
    print(f"\n[PnP求解] Ref点网格位置: ({ref_row}, {ref_col}) -> 3D原点(0,0,0)")
    print(f"[PnP求解] 3D范围: X=[{np.min(object_points[:,0]):.3f}, {np.max(object_points[:,0]):.3f}]m, "
          f"Y=[{np.min(object_points[:,1]):.3f}, {np.max(object_points[:,1]):.3f}]m")
    
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None, None, None, None, None
    
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec,
                                           camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    
    errors = np.linalg.norm(image_points - projected_points, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    R_camera_to_board, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R_camera_to_board[0, 0]**2 + R_camera_to_board[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        beta = np.arctan2(R_camera_to_board[2, 1], R_camera_to_board[2, 2])
        alpha = np.arctan2(-R_camera_to_board[2, 0], sy)
        gamma = np.arctan2(R_camera_to_board[1, 0], R_camera_to_board[0, 0])
    else:
        beta = np.arctan2(-R_camera_to_board[1, 2], R_camera_to_board[1, 1])
        alpha = np.arctan2(-R_camera_to_board[2, 0], sy)
        gamma = 0
    
    delta_x = tvec[0, 0]
    delta_y = tvec[1, 0]
    delta_z = tvec[2, 0]
    
    transform_array = np.array([delta_x, delta_y, delta_z, gamma, alpha, beta])
    euler_angles_deg = {
        'roll': np.degrees(beta),
        'pitch': np.degrees(alpha),
        'yaw': np.degrees(gamma)
    }
    
    return transform_array, euler_angles_deg, mean_error, max_error, (rvec, tvec)


def detect_grid_with_apriltag(image, camera_matrix, dist_coeffs, max_distance=100.0):
    """主检测函数 - 融合Version 4和Version 8 (集成方案三：木匠拉线法)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 检测 AprilTag
    apriltag_info = detect_apriltag(gray)
    if apriltag_info is None:
        return {'success': False, 'error': 'No AprilTag detected'}
    
    # 2. 检测所有 Blob
    blob_points, keypoints = detect_blobs(gray)
    if len(blob_points) == 0:
        return {'success': False, 'error': 'No blobs detected'}
    
    # 3. 过滤干扰点
    filtered_blob_points, excluded_indices = filter_blobs_exclude_apriltag(
        blob_points, apriltag_info
    )
    filtered_blob_points, outlier_indices = filter_blobs_by_grid_pattern(
        filtered_blob_points, apriltag_info
    )
    
    # 兜底策略
    if len(filtered_blob_points) < 100:
        print(f"  ⚠️  过滤后点数不足，使用保守策略")
        filtered_blob_points, _ = filter_blobs_exclude_apriltag(blob_points, apriltag_info)
    
# 4. 【关键修改】不再找单个角点，而是直接拟合所有角点
    # 这一步会非常稳健，因为利用了所有点的信息
    virtual_corners = find_all_virtual_corners(filtered_blob_points)
    
    if virtual_corners is None:
        # 如果拟合失败（比如点太少），回退到原来的逻辑
        print("全局拟合失败，回退到普通逻辑...")
        corners_dict, _ = find_four_corner_blobs(filtered_blob_points)
        # ... (这里可以写你原来的逻辑作为备用)
        return {'success': False, 'error': 'Corner fitting failed'}
    
    # 5. 保存拟合出的角点用于后续计算和显示
    corners_dict = virtual_corners
    result_corners_dict = corners_dict.copy()
    
    # 6. 确定 Ref 点 (根据 AprilTag 找最近的那个拟合角点)
    # 我们遍历 4 个拟合出的角点，看谁离 Tag 最近
    min_dist = float('inf')
    corner_name = 'top_right' # 默认
    corner_blob = virtual_corners['top_right'] # 默认
    
    tag_center = apriltag_info['center']
    
    for name, pt in virtual_corners.items():
        dist = np.linalg.norm(pt - tag_center)
        if dist < min_dist:
            min_dist = dist
            corner_name = name
            corner_blob = pt
            
    print(f"[Ref点] 确定离Tag最近的角点是: {corner_name}")
    
    # 7. 计算间距 (用拟合的角点算，非常准)
    spacing_px = estimate_spacing_from_corners(corners_dict, pattern_size=(15, 15))
    
    # 8. 【关键修改】直接用 Homography 构建网格
    # 此时生成的网格，天然带有透视效果，不会飞，不会断！
    coarse_grid_points = build_grid_by_homography(virtual_corners, pattern_size=(15, 15))
    
    # 9. 初始匹配
    matched_corners, matched_indices = match_blobs_to_grid(
        filtered_blob_points, coarse_grid_points, max_distance=max_distance
    )
    
    if matched_corners is None or len(matched_corners) < 4:
        return {'success': False, 'error': 'Insufficient matched points'}
    
    # 10. 单应性优化
    theoretical_grid_points, H = refine_grid_with_homography(
        coarse_grid_points, matched_indices, matched_corners,
        pattern_size=(15, 15), ref_grid_pos=get_ref_grid_position(corner_name), ref_point=corner_blob
    )
    
    if H is None:
        return {'success': False, 'error': 'Homography failed'}
    
    # 11. 智能补全
    search_radius = spacing_px * 0.5
    final_grid_points, status_mask, n_found, n_interp, final_matched_blobs, final_matched_indices = match_and_fill_missing_points(
        theoretical_grid_points, filtered_blob_points, 
        search_radius=search_radius, pattern_size=(15, 15)
    )
    
    # 12. 强制锚点逻辑
    ref_grid_pos = get_ref_grid_position(corner_name)
    r_ref, c_ref = ref_grid_pos
    
    if virtual_corners is None:
        final_grid_points[r_ref, c_ref] = corner_blob
        status_mask[r_ref, c_ref] = 1
        print(f"[强制锚点] 使用物理 Ref 点覆盖网格")
    else:
        print(f"[虚拟锚点] 保持计算出的虚拟坐标作为 PnP 原点")

    # 13. PnP 解算
    result = solve_pnp_with_ref_origin(
        final_grid_points, ref_grid_pos, camera_matrix, dist_coeffs
    )
    
    if result[0] is None:
        return {'success': False, 'error': 'PnP solving failed'}
    
    transform_array, euler_angles_deg, mean_error, max_error, pose = result

    # 14. 返回结果
    return {
        'success': True,
        'final_grid_points': final_grid_points,
        'theoretical_grid_points': theoretical_grid_points,
        'status_mask': status_mask,
        'apriltag_info': apriltag_info,
        'corner_name': corner_name,
        'ref_grid_pos': ref_grid_pos,
        'corners_dict': result_corners_dict,
        'final_matched_blobs': final_matched_blobs,
        'final_matched_indices': final_matched_indices,
        'spacing_px': spacing_px,
        'all_keypoints': keypoints,
        'transform_array': transform_array,
        'euler_angles_deg': euler_angles_deg,
        'mean_error': mean_error,
        'max_error': max_error,
        'pose': pose,
        'n_found': n_found,
        'n_interpolated': n_interp,
        'filtered_blobs': filtered_blob_points,
        'result': result
    }


import cv2
import numpy as np

def visualize_result(image, result, camera_matrix, dist_coeffs, image_name):
    """可视化 - 红色实心点版"""
    vis = image.copy()

    # 1. 绘制所有原始blob点（绿色空心圆圈）
    if 'all_keypoints' in result:
        for kp in result['all_keypoints']:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2) + 2
            cv2.circle(vis, (x, y), radius, (0, 255, 0), 1)

    # 2. 绘制过滤后的点（白色十字）
    if 'filtered_blobs' in result:
        filtered_points = result['filtered_blobs']
        for pt in filtered_points:
            center = tuple(pt.astype(int))
            cv2.drawMarker(vis, center, (255, 255, 255), 
                         cv2.MARKER_CROSS, 20, 2)
    
    # 3. 绘制AprilTag
    if 'apriltag_info' in result:
        tag_info = result['apriltag_info']
        tag_corners = tag_info['corners'].astype(int)
        tag_center = tag_info['center'].astype(int)
        
        cv2.polylines(vis, [tag_corners], True, (0, 255, 0), 3)
        cv2.putText(vis, f"ID:{tag_info['tag_id']}", 
                    tuple(tag_center + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # 4. 绘制四个角点（蓝色方块）
    if 'corners_dict' in result:
        corners_dict = result['corners_dict']
        for corner_name, corner_pt in corners_dict.items():
            pt = tuple(corner_pt.astype(int))
            cv2.rectangle(vis, (pt[0]-8, pt[1]-8), (pt[0]+8, pt[1]+8), (255, 0, 0), -1)
            # 简化标签，防止遮挡
            label_offset = {'top_left': (-40, -15), 'top_right': (15, -15),
                          'bottom_left': (-40, 25), 'bottom_right': (15, 25)}
            offset = label_offset.get(corner_name, (15, -15))
            cv2.putText(vis, corner_name, (pt[0]+offset[0], pt[1]+offset[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # 5. 绘制最终匹配点（这个可以先保留，作为参考）
    if 'final_matched_blobs' in result and len(result['final_matched_blobs']) > 0:
        final_matched_blobs = result['final_matched_blobs']
        for i in range(len(final_matched_blobs)):
            pt = tuple(final_matched_blobs[i].astype(int))
            # 这里的匹配点画小一点，以免盖住下面的网格点
            cv2.circle(vis, pt, 2, (255, 0, 255), -1) 
    
    # =================================================================
    # 🔥🔥🔥 核心修改区：绘制所有 grid_points 为红色实心点 🔥🔥🔥
    # =================================================================
    if 'final_grid_points' in result:
        grid = result['final_grid_points']
        rows, cols = grid.shape[:2]
        
        for r in range(rows):
            for c in range(cols):
                pt = tuple(grid[r, c].astype(int))
                
                # 1. 绘制红色实心点 (Red Solid Point)
                # 半径设为 4，颜色为 (0, 0, 255) 红色
                cv2.circle(vis, pt, 4, (0, 0, 255), -1)
                
                # 2. (可选) 如果是插值补全的点，加个白圈区分一下，方便调试
                # 这样既有红色实心点，又能看出哪些是算的，哪些是匹配的
                if 'status_mask' in result and result['status_mask'][r, c] == 0:
                    cv2.circle(vis, pt, 6, (255, 255, 255), 1)
    # =================================================================
    
    # 7. 标注Ref点 (橙色大圈)
    if 'ref_grid_pos' in result and 'final_grid_points' in result:
        grid = result['final_grid_points']
        r_ref, c_ref = result['ref_grid_pos']
        ref_pt = tuple(grid[r_ref, c_ref].astype(int))
        
        cv2.circle(vis, ref_pt, 20, (0, 165, 255), 3)
        cv2.putText(vis, "Ref", (ref_pt[0]+25, ref_pt[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        if 'apriltag_info' in result:
             cv2.line(vis, tuple(result['apriltag_info']['center'].astype(int)), 
                        ref_pt, (255, 0, 255), 2, cv2.LINE_AA)
    
    # 8. 绘制坐标轴
    if 'pose' in result:
        axis_length = 0.1
        axis_3d = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, -axis_length]
        ])
        
        rvec, tvec = result['pose']
        try:
            imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = imgpts.reshape(-1, 2).astype(int)
            
            origin = tuple(imgpts[0])
            cv2.arrowedLine(vis, origin, tuple(imgpts[1]), (255, 0, 255), 5, tipLength=0.3)
            cv2.arrowedLine(vis, origin, tuple(imgpts[2]), (0, 0, 255), 5, tipLength=0.3)
            cv2.arrowedLine(vis, origin, tuple(imgpts[3]), (255, 0, 0), 5, tipLength=0.3)
            
            cv2.putText(vis, "X", tuple(imgpts[1] + [15, 0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            cv2.putText(vis, "Y", tuple(imgpts[2] + [15, 0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(vis, "Z", tuple(imgpts[3] + [15, 0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        except Exception:
            pass
    
    # 9. 信息面板
    if 'euler_angles_deg' in result:
        euler = result['euler_angles_deg']
        matched_count = len(result['final_matched_blobs']) if 'final_matched_blobs' in result else 0
        error_val = result.get('mean_error', 0.0)
        
        info_lines = [
            f"Image: {image_name}",
            f"Roll: {euler['roll']:+.2f} | Pitch: {euler['pitch']:+.2f} | Yaw: {euler['yaw']:+.2f}",
            f"Error: Avg={error_val:.3f}px | Matched: {matched_count}"
        ]
        
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (650, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        
        for i, line in enumerate(info_lines):
            cv2.putText(vis, line, (10, 30 + i * 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis


def process_all_images():
    """批量处理"""
    data_folder = Path('data')
    output_folder = Path('outputs/newpy/images')
    camera_yaml = '/home/eureka/tilt_checker2/-AprilTag-/config/camera_info.yaml'
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("加载相机参数...")
    camera_matrix, dist_coeffs = load_camera_params(camera_yaml)
    
    if camera_matrix is None:
        return
    
    image_files = sorted(list(data_folder.glob('*.png')) + list(data_folder.glob('*.jpg')))
    if len(image_files) == 0:
        print(f"❌ 在 {data_folder} 中未找到图片")
        return
    
    all_results = []
    success_count = 0
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"处理第 {idx}/{len(image_files)} 张图片: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        result = detect_grid_with_apriltag(image_undistorted, camera_matrix, dist_coeffs)
        
        if not result['success']:
            print(f"❌ 检测失败: {result.get('error', 'Unknown')}")
            continue
        
        success_count += 1
        print(f"✅ 成功! 误差: Avg={result['mean_error']:.3f}px | Ref: {result['corner_name']}")
        
        vis_image = visualize_result(image_undistorted, result, camera_matrix,
                                     dist_coeffs, img_path.name)
        
        output_name = img_path.stem + '_result.png'
        cv2.imwrite(str(output_folder / output_name), vis_image)
        
        all_results.append({
            'image_name': img_path.name,
            'transform': result['transform_array'],
            'mean_error': result['mean_error'],
            'max_error': result['max_error']
        })
    
    print(f"\n{'='*80}")
    print(f"批量处理完成！成功: {success_count}/{len(image_files)}")
    
    if len(all_results) > 0:
        result_file = Path('Result_array.txt')
        with open(result_file, 'w') as f:
            for res in all_results:
                t = res['transform']
                line = f"{res['image_name']} [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}, {t[3]:.6f}, {t[4]:.6f}, {t[5]:.6f}] {res['mean_error']:.4f}\n"
                f.write(line)
        print(f"💾 结果已保存到: {result_file}")


if __name__ == '__main__':
    process_all_images()