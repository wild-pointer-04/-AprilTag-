"""
改进版圆点网格检测
包含自适应参数、智能搜索、改进的亚像素定位等优化
"""
import cv2
import numpy as np
from src.utils import build_blob_detector


def build_adaptive_blob_detector(gray, base_params=None):
    """兼容旧接口，统一使用 utils.build_blob_detector() 的参数。"""
    return build_blob_detector()


def refine_circle_center_improved(gray, initial_pt, radius_estimate=None):
    """
    改进的圆点中心定位（基于质心计算）
    
    Args:
        gray: 灰度图像
        initial_pt: 初始点坐标 [x, y]
        radius_estimate: 估计的圆点半径（可选）
    
    Returns:
        精化后的中心坐标
    """
    # 确保正确提取标量值
    initial_pt = np.asarray(initial_pt).flatten()
    x, y = int(float(initial_pt[0])), int(float(initial_pt[1]))
    
    # 如果没有半径估计，使用固定窗口大小
    if radius_estimate is None:
        r = 10
    else:
        r = int(radius_estimate * 1.5)
    
    # 提取ROI
    y_min = max(0, y - r)
    y_max = min(gray.shape[0], y + r + 1)
    x_min = max(0, x - r)
    x_max = min(gray.shape[1], x + r + 1)
    
    roi = gray[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return initial_pt
    
    # 自适应阈值
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 计算质心
    M = cv2.moments(binary)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        # 转换回原图坐标
        refined_x = cx + x_min
        refined_y = cy + y_min
        return np.array([refined_x, refined_y], dtype=np.float32)
    else:
        return initial_pt


def refine_improved(gray, pts, use_weighted_centroid=True):
    """
    改进的亚像素精化
    
    Args:
        gray: 灰度图像
        pts: 初始点集
        use_weighted_centroid: 是否使用加权质心方法
    """
    if use_weighted_centroid:
        # 方法1: 加权质心（对圆点更精确）
        # 确保 pts 是正确格式
        pts = np.asarray(pts).reshape(-1, 2)
        refined = []
        for pt in pts:
            refined.append(refine_circle_center_improved(gray, pt))
        return np.array(refined, dtype=np.float32)
    else:
        # 方法2: 传统 cornerSubPix（对网格角点更精确）
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        return cv2.cornerSubPix(g, pts.astype(np.float32), (5, 5), (-1, -1), crit)


def smart_auto_search(gray, rows_range=(8, 28), cols_range=(8, 28), max_attempts=100, timeout_seconds=20.0):
    """
    智能自动搜索，先估计网格尺寸再搜索
    
    Args:
        gray: 灰度图像
        rows_range: 行数范围
        cols_range: 列数范围
        max_attempts: 最大尝试次数（避免无限循环）
        timeout_seconds: 超时时间（秒）
    
    Returns:
        (成功标志, 精化后的点集, (rows, cols, symmetric), keypoints)
    """
    import time
    start_time = time.time()
    
    # 1. 先检测所有圆点，估算网格尺寸
    det = build_adaptive_blob_detector(gray)
    keypoints = det.detect(gray)
    num_points = len(keypoints)
    
    print(f"[DEBUG] Blob检测: 找到 {num_points} 个候选圆点")
    
    if num_points == 0:
        print("[DEBUG] 错误: 未检测到任何圆点候选")
        return False, None, None, []
    
    # 如果检测到的点太少，直接返回失败
    min_required_points = rows_range[0] * cols_range[0]
    if num_points < min_required_points * 0.5:
        print(f"[DEBUG] 错误: 检测到的点数({num_points})远少于最小要求({min_required_points})")
        print(f"[DEBUG] 建议: 1) 调整 blob 检测参数 2) 改善光照条件 3) 检查标定板是否完整可见")
        return False, None, None, keypoints
    
    # 2. 估算可能的网格尺寸
    estimated_size = int(np.sqrt(num_points))
    print(f"[DEBUG] 估算网格尺寸: {estimated_size}×{estimated_size} (基于 {num_points} 个点)")
    
    # 3. 在估计值附近优先搜索（对称网格）
    center = estimated_size
    search_range = max(2, estimated_size // 4)
    
    # 优先搜索范围
    priority_rows = range(
        max(rows_range[0], center - search_range),
        min(rows_range[1] + 1, center + search_range + 1)
    )
    priority_cols = range(
        max(cols_range[0], center - search_range),
        min(cols_range[1] + 1, center + search_range + 1)
    )
    
    attempt_count = 0
    
    # 尝试对称网格
    for symmetric in (True, False):
        # 优先搜索
        for r in priority_rows:
            for c in priority_cols:
                # 检查超时
                if time.time() - start_time > timeout_seconds:
                    print(f"[DEBUG] 搜索超时 ({timeout_seconds}秒)，已尝试 {attempt_count} 次")
                    return False, None, None, keypoints
                
                # 检查最大尝试次数
                attempt_count += 1
                if attempt_count > max_attempts:
                    print(f"[DEBUG] 达到最大尝试次数 ({max_attempts})，停止搜索")
                    return False, None, None, keypoints
                
                if r * c <= num_points * 1.2:  # 允许少量缺失（20%）
                    ok, centers, kps = try_find_adaptive(gray, r, c, symmetric=symmetric)
                    if ok:
                        print(f"[DEBUG] ✅ 成功匹配: {r}×{c}, symmetric={symmetric} (尝试了 {attempt_count} 次)")
                        return True, refine_improved(gray, centers), (r, c, symmetric), kps
        
        # 完整搜索（如果优先搜索失败）- 限制范围避免过长时间
        if attempt_count < max_attempts // 2:  # 只有在还有足够尝试次数时才进行完整搜索
            print(f"[DEBUG] 优先搜索失败，尝试有限的完整搜索 (symmetric={symmetric})...")
            # 限制完整搜索的范围
            limited_rows = range(
                max(rows_range[0], center - search_range - 2),
                min(rows_range[1] + 1, center + search_range + 3)
            )
            limited_cols = range(
                max(cols_range[0], center - search_range - 2),
                min(cols_range[1] + 1, center + search_range + 3)
            )
            
            for r in limited_rows:
                for c in limited_cols:
                    # 检查超时
                    if time.time() - start_time > timeout_seconds:
                        print(f"[DEBUG] 搜索超时 ({timeout_seconds}秒)，已尝试 {attempt_count} 次")
                        return False, None, None, keypoints
                    
                    # 检查最大尝试次数
                    attempt_count += 1
                    if attempt_count > max_attempts:
                        print(f"[DEBUG] 达到最大尝试次数 ({max_attempts})，停止搜索")
                        return False, None, None, keypoints
                    
                    if r * c <= num_points * 1.2:
                        ok, centers, kps = try_find_adaptive(gray, r, c, symmetric=symmetric)
                        if ok:
                            print(f"[DEBUG] ✅ 成功匹配: {r}×{c}, symmetric={symmetric} (尝试了 {attempt_count} 次)")
                            return True, refine_improved(gray, centers), (r, c, symmetric), kps
    
    elapsed = time.time() - start_time
    print(f"[DEBUG] ❌ 自动搜索失败: 尝试了 {attempt_count} 次组合，耗时 {elapsed:.2f}秒")
    print(f"[DEBUG] 可能原因:")
    print(f"[DEBUG]   1) 网格结构不完整 - 边缘圆点缺失或遮挡")
    print(f"[DEBUG]   2) 透视畸变过大 - 相机角度太倾斜")
    print(f"[DEBUG]   3) Blob检测参数不合适 - 检测到太多噪声点或漏检")
    print(f"[DEBUG]   4) 光照条件不佳 - 对比度不足或反光")
    print(f"[DEBUG] 建议:")
    print(f"[DEBUG]   1) 调整相机位置，使标定板更正对相机")
    print(f"[DEBUG]   2) 改善光照，确保圆点清晰可见")
    print(f"[DEBUG]   3) 检查标定板是否完整在视野内")
    print(f"[DEBUG]   4) 尝试调整 blob 检测参数（在 utils.py 中）")
    
    return False, None, None, keypoints


def try_find_adaptive(gray, rows, cols, symmetric=True, use_preprocessing=False):
    """
    使用自适应参数尝试检测网格，包含多种策略
    
    Args:
        gray: 灰度图像
        rows: 行数
        cols: 列数
        symmetric: 是否对称网格
        use_preprocessing: 是否使用图像预处理
    
    Returns:
        (成功标志, 检测到的点集, blob关键点)
    """
    # 策略1: 标准检测
    ok, centers, keypoints = _try_single_strategy(gray, rows, cols, symmetric, use_preprocessing)
    if ok:
        return ok, centers, keypoints
    
    # 策略2: 增强预处理
    if not use_preprocessing:
        ok, centers, keypoints = _try_single_strategy(gray, rows, cols, symmetric, True)
        if ok:
            return ok, centers, keypoints
    
    # 策略3: 透视校正预处理
    gray_corrected = preprocess_for_detection(gray, enhance_contrast=True, denoise=True, correct_perspective=True)
    ok, centers, keypoints = _try_single_strategy(gray_corrected, rows, cols, symmetric, False)
    if ok:
        return ok, centers, keypoints
    
    # 策略4: 放宽检测参数
    ok, centers, keypoints = _try_relaxed_detection(gray, rows, cols, symmetric)
    if ok:
        return ok, centers, keypoints
    
    return False, None, keypoints


def _try_single_strategy(gray, rows, cols, symmetric, use_preprocessing):
    """单一策略检测"""
    flags = cv2.CALIB_CB_CLUSTERING
    flags |= (cv2.CALIB_CB_SYMMETRIC_GRID if symmetric else cv2.CALIB_CB_ASYMMETRIC_GRID)
    
    # 可选的图像预处理
    if use_preprocessing:
        gray_processed = preprocess_for_detection(gray, enhance_contrast=True, denoise=False)
    else:
        gray_processed = gray
    
    det = build_adaptive_blob_detector(gray_processed)
    keypoints = det.detect(gray_processed)
    
    # 尝试网格匹配
    ok, centers = cv2.findCirclesGrid(gray_processed, (cols, rows), flags=flags, blobDetector=det)
    
    return ok, centers, keypoints


def _try_relaxed_detection(gray, rows, cols, symmetric):
    """使用更宽松参数的检测"""
    # 创建更宽松的blob检测器
    p = cv2.SimpleBlobDetector_Params()
    
    # 更宽松的参数
    p.filterByColor = True
    p.blobColor = 0
    
    p.filterByArea = True
    p.minArea = 30           # 更小的最小面积
    p.maxArea = 15000        # 更大的最大面积
    
    p.filterByCircularity = True
    p.minCircularity = 0.05  # 极低的圆度要求
    p.maxCircularity = 1.0
    
    p.filterByInertia = True
    p.minInertiaRatio = 0.01 # 极低的惯性比
    
    p.filterByConvexity = False
    
    p.minThreshold = 1       # 极低的阈值
    p.maxThreshold = 250
    p.thresholdStep = 5
    
    p.minDistBetweenBlobs = 1.0
    
    relaxed_detector = cv2.SimpleBlobDetector_create(p)
    
    flags = cv2.CALIB_CB_CLUSTERING
    flags |= (cv2.CALIB_CB_SYMMETRIC_GRID if symmetric else cv2.CALIB_CB_ASYMMETRIC_GRID)
    
    keypoints = relaxed_detector.detect(gray)
    ok, centers = cv2.findCirclesGrid(gray, (cols, rows), flags=flags, blobDetector=relaxed_detector)
    
    return ok, centers, keypoints


def evaluate_detection_quality(centers, rows, cols):
    """
    评估检测质量（用于多尺度检测中选择最佳结果）
    
    Args:
        centers: 检测到的点集
        rows: 行数
        cols: 列数
    
    Returns:
        质量分数（0-1）
    """
    if centers is None or len(centers) == 0:
        return 0.0
    
    expected_points = rows * cols
    detected_points = len(centers)
    
    # 点数完整性
    completeness = detected_points / expected_points
    
    # 网格一致性（计算相邻点间距的方差，越小越好）
    if detected_points >= 4:
        centers_2d = centers.reshape(-1, 2)
        # 计算相邻点间距
        distances = []
        for i in range(min(rows-1, detected_points//cols-1)):
            for j in range(min(cols-1, detected_points%cols)):
                idx = i * cols + j
                if idx + 1 < len(centers_2d):
                    dist = np.linalg.norm(centers_2d[idx+1] - centers_2d[idx])
                    distances.append(dist)
        
        if len(distances) > 1:
            consistency = 1.0 / (1.0 + np.std(distances) / np.mean(distances))
        else:
            consistency = 0.5
    else:
        consistency = 0.5
    
    # 综合分数
    score = completeness * 0.7 + consistency * 0.3
    return score


def preprocess_for_detection(gray, enhance_contrast=True, denoise=True, correct_perspective=False):
    """
    预处理图像以提高检测率
    
    Args:
        gray: 原始灰度图像
        enhance_contrast: 是否增强对比度
        denoise: 是否去噪
        correct_perspective: 是否尝试透视校正（实验性）
    
    Returns:
        预处理后的图像
    """
    enhanced = gray.copy()
    
    # 1. 去噪（可选）
    if denoise:
        # 使用双边滤波：保留边缘的同时去除噪声
        enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 2. 对比度增强（可选）
    if enhance_contrast:
        # CLAHE（对比度受限的自适应直方图均衡）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
    
    # 3. 透视校正预处理（实验性）
    if correct_perspective:
        enhanced = apply_perspective_correction_hint(enhanced)
    
    return enhanced


def apply_perspective_correction_hint(gray):
    """
    基于图像特征的轻量级透视校正提示
    不做完整的透视校正，只是改善检测条件
    """
    # 使用自适应阈值突出圆点
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # 轻微的形态学操作，连接断裂的圆点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    
    # 与原图融合，保持灰度信息
    enhanced = cv2.addWeighted(gray, 0.7, 255 - adaptive, 0.3, 0)
    
    return enhanced


# 兼容性：保持与原版相同的接口
def try_find(gray, rows, cols, symmetric=True):
    """兼容原版接口"""
    ok, centers, keypoints = try_find_adaptive(gray, rows, cols, symmetric)
    return ok, centers, keypoints


def refine(gray, pts):
    """兼容原版接口"""
    return refine_improved(gray, pts, use_weighted_centroid=False)


def auto_search(gray, rows_range=(8, 28), cols_range=(8, 28)):
    """兼容原版接口，但使用改进的智能搜索"""
    result = smart_auto_search(gray, rows_range, cols_range)
    if len(result) == 4:  # 成功时返回 (True, corners, meta, keypoints)
        return result[0], result[1], result[2], result[3]
    else:  # 失败时返回 (False, None, None, keypoints)
        return result[0], result[1], result[2], result[3]

