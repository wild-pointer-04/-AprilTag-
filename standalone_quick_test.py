#!/usr/bin/env python3
"""
独立测试脚本 - 不依赖外部模块
直接测试不同参数配置
"""

import cv2
import numpy as np

try:
    from pupil_apriltags import Detector
    USING_PUPIL = True
except ImportError:
    import apriltag
    USING_PUPIL = False


def simple_greedy_matching(cost_matrix, threshold=1e6):
    """
    简单的贪心匹配算法
    每次选择当前最小cost的未匹配对
    """
    n_rows, n_cols = cost_matrix.shape
    
    # 创建所有可能的匹配及其cost
    matches = []
    for i in range(n_rows):
        for j in range(n_cols):
            if cost_matrix[i, j] < threshold:
                matches.append((cost_matrix[i, j], i, j))
    
    # 按cost排序
    matches.sort()
    
    # 贪心选择
    used_rows = set()
    used_cols = set()
    result_rows = []
    result_cols = []
    
    for cost, i, j in matches:
        if i not in used_rows and j not in used_cols:
            result_rows.append(i)
            result_cols.append(j)
            used_rows.add(i)
            used_cols.add(j)
    
    return np.array(result_rows), np.array(result_cols)


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
    
    if USING_PUPIL:
        corners = np.array(detection.corners, dtype=np.float64)
        center = np.array(detection.center, dtype=np.float64)
        tag_id = detection.tag_id
    else:
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
    params.minArea = 100
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.6
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_image)
    
    blob_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return blob_points, keypoints


def estimate_grid(apriltag_info, circle_spacing, apriltag_size, 
                 apriltag_position, image_shape, pattern_size=(15, 15)):
    """估算网格位置 - 彻底修正Y轴方向"""
    h, w = image_shape[:2]
    tag_center = apriltag_info['center']
    tag_corners = apriltag_info['corners']
    
    print(f"\n[调试] AprilTag中心: {tag_center}")
    print(f"[调试] AprilTag角点:")
    for i, corner in enumerate(tag_corners):
        print(f"  角点{i}: {corner} (Y={corner[1]:.1f})")
    
    # 关键修正：正确识别哪个角点在上，哪个在下
    # 先找出Y坐标最小（最上方）和最大（最下方）的角点
    y_coords = [corner[1] for corner in tag_corners]
    top_idx = np.argmin(y_coords)
    bottom_idx = np.argmax(y_coords)
    
    print(f"[调试] 最上方角点: {top_idx} (Y={y_coords[top_idx]:.1f})")
    print(f"[调试] 最下方角点: {bottom_idx} (Y={y_coords[bottom_idx]:.1f})")
    
    # 根据AprilTag的标准定义：
    # 如果检测正确：角点0左上, 角点1右上, 角点2右下, 角点3左下
    # 但实际可能被旋转或翻转
    
    # 我们需要找到：
    # 1. 哪两个角点在上方（Y较小）
    # 2. 哪两个角点在下方（Y较大）
    # 3. 在上方的两个点中，哪个在左（X较小），哪个在右（X较大）
    
    # 按Y坐标排序
    corners_with_idx = [(i, tag_corners[i]) for i in range(4)]
    corners_with_idx.sort(key=lambda x: x[1][1])  # 按Y排序
    
    # 上方两个点
    top_two = corners_with_idx[:2]
    # 下方两个点  
    bottom_two = corners_with_idx[2:]
    
    # 在上方两个点中，按X排序确定左右
    top_two.sort(key=lambda x: x[1][0])
    top_left = top_two[0][1]
    top_right = top_two[1][1]
    
    # 在下方两个点中，按X排序确定左右
    bottom_two.sort(key=lambda x: x[1][0])
    bottom_left = bottom_two[0][1]
    bottom_right = bottom_two[1][1]
    
    print(f"[调试] 重新识别的角点:")
    print(f"  左上: {top_left}")
    print(f"  右上: {top_right}")
    print(f"  左下: {bottom_left}")
    print(f"  右下: {bottom_right}")
    
    # 计算方向向量
    tag_x_vec = top_right - top_left  # X轴：从左上到右上
    tag_y_vec = bottom_left - top_left  # Y轴：从左上到左下（向下）
    
    tag_x_len = np.linalg.norm(tag_x_vec)
    tag_y_len = np.linalg.norm(tag_y_vec)
    
    unit_x = tag_x_vec / tag_x_len
    unit_y = tag_y_vec / tag_y_len
    
    print(f"[调试] X方向向量: {unit_x} (长度={tag_x_len:.2f}px)")
    print(f"[调试] Y方向向量: {unit_y} (长度={tag_y_len:.2f}px)")
    
    # 验证Y轴确实向下
    if unit_y[1] < 0:
        print(f"[错误] Y轴仍然向上！这不应该发生。")
        print(f"       top_left: {top_left}")
        print(f"       bottom_left: {bottom_left}")
    else:
        print(f"[正确] Y轴方向向下 ✓")
    
    # 计算比例
    pixel_per_meter = (tag_x_len + tag_y_len) / (2.0 * apriltag_size)
    circle_spacing_px = circle_spacing * pixel_per_meter
    
    print(f"[调试] 像素/米比例: {pixel_per_meter:.2f} px/m")
    print(f"[调试] 圆点间距: {circle_spacing_px:.2f} px")
    
    # 使用重新识别的右上角作为参考点（AprilTag在这里）
    # 注意：tag_center 可能不准确，我们直接用 top_right
    reference_point = top_right
    
    if apriltag_position == 'right_top_inside':
        # AprilTag替代了(row=0, col=14)位置的圆点
        offset_cols = 14
        offset_rows = 0
    else:
        # AprilTag在外部
        offset_cols = 14.5
        offset_rows = 0
    
    # 计算网格左上角(0,0)的位置
    # 从右上角向左移动offset_cols，向下移动0
    grid_origin = reference_point - unit_x * offset_cols * circle_spacing_px
    
    print(f"[调试] 参考点（右上）: {reference_point}")
    print(f"[调试] 网格原点(0,0): {grid_origin}")
    print(f"[调试] 从参考点到原点: 向左{offset_cols}格")
    
    # 生成网格点
    grid_points = np.zeros((pattern_size[1], pattern_size[0], 2), dtype=np.float32)
    valid_mask = np.zeros((pattern_size[1], pattern_size[0]), dtype=bool)
    
    margin = 20
    for row in range(pattern_size[1]):
        for col in range(pattern_size[0]):
            # 从原点向右移动col格，向下移动row格
            point = grid_origin + unit_x * col * circle_spacing_px + unit_y * row * circle_spacing_px
            grid_points[row, col] = point
            
            if (margin <= point[0] < w - margin and 
                margin <= point[1] < h - margin):
                valid_mask[row, col] = True
    
    # 打印关键点
    print(f"[调试] 关键网格点位置:")
    print(f"  (0,0)左上角: {grid_points[0, 0]}")
    print(f"  (0,14)右上角: {grid_points[0, 14]}")
    print(f"  (14,0)左下角: {grid_points[14, 0]}")
    print(f"  (14,14)右下角: {grid_points[14, 14]}")
    print(f"  (7,7)中心: {grid_points[7, 7]}")
    
    print(f"[调试] 有效网格点: {np.sum(valid_mask)}/{pattern_size[0]*pattern_size[1]}")
    
    return grid_points, valid_mask


def match_blobs_to_grid(blob_points, grid_points, valid_mask, max_distance=25.0):
    """匹配blob到网格"""
    rows, cols = grid_points.shape[:2]
    
    # 收集有效网格点
    valid_grids = []
    valid_indices = []
    for row in range(rows):
        for col in range(cols):
            if valid_mask[row, col]:
                valid_grids.append(grid_points[row, col])
                valid_indices.append((row, col))
    
    if len(valid_grids) == 0 or len(blob_points) == 0:
        return None, None, np.zeros((rows, cols), dtype=bool)
    
    valid_grids = np.array(valid_grids)
    valid_indices = np.array(valid_indices)
    
    # 计算距离矩阵
    n_grids = len(valid_grids)
    n_blobs = len(blob_points)
    
    cost_matrix = np.zeros((n_grids, n_blobs), dtype=np.float32)
    
    for i in range(n_grids):
        for j in range(n_blobs):
            dist = np.linalg.norm(valid_grids[i] - blob_points[j])
            if dist <= max_distance:
                cost_matrix[i, j] = dist
            else:
                cost_matrix[i, j] = 1e6
    
    # 贪心匹配
    grid_indices, blob_indices = simple_greedy_matching(cost_matrix, threshold=max_distance)
    
    # 构建结果
    matched_corners = []
    matched_indices = []
    match_mask = np.zeros((rows, cols), dtype=bool)
    
    for grid_idx, blob_idx in zip(grid_indices, blob_indices):
        if cost_matrix[grid_idx, blob_idx] < max_distance:
            row, col = valid_indices[grid_idx]
            matched_corners.append(blob_points[blob_idx])
            matched_indices.append([row, col])
            match_mask[row, col] = True
    
    if len(matched_corners) == 0:
        return None, None, match_mask
    
    matched_corners = np.array(matched_corners, dtype=np.float32).reshape(-1, 1, 2)
    matched_indices = np.array(matched_indices, dtype=np.int32)
    
    return matched_corners, matched_indices, match_mask


def test_config(image, config):
    """测试一个配置"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测AprilTag
    apriltag_info = detect_apriltag(gray)
    if apriltag_info is None:
        return None
    
    # 检测blob
    blob_points, keypoints = detect_blobs(gray)
    if len(blob_points) == 0:
        return None
    
    # 估算网格
    grid_points, valid_mask = estimate_grid(
        apriltag_info,
        config['circle_spacing'],
        config['apriltag_size'],
        config['apriltag_position'],
        image.shape
    )
    
    # 匹配 - 使用配置中的阈值
    max_dist = config.get('max_distance', 25.0)
    matched_corners, matched_indices, match_mask = match_blobs_to_grid(
        blob_points, grid_points, valid_mask, max_distance=max_dist
    )
    
    if matched_corners is None:
        return None
    
    valid_count = np.sum(valid_mask)
    match_count = len(matched_corners)
    
    return {
        'success': True,
        'match_count': match_count,
        'valid_count': valid_count,
        'match_rate': match_count / valid_count * 100 if valid_count > 0 else 0,
        'matched_corners': matched_corners,
        'matched_indices': matched_indices,
        'grid_points': grid_points,
        'valid_mask': valid_mask,
        'match_mask': match_mask,
        'blob_points': blob_points,
        'keypoints': keypoints,
        'apriltag_info': apriltag_info,
        'max_distance': max_dist
    }


def visualize_result(image, result, config_name):
    """可视化结果"""
    vis = image.copy()
    
    # 绘制blob（绿色圆圈）
    for kp in result['keypoints']:
        pt = (int(kp.pt[0]), int(kp.pt[1]))
        cv2.circle(vis, pt, int(kp.size/2), (0, 255, 0), 2)
    
    # 绘制网格点
    grid_points = result['grid_points']
    valid_mask = result['valid_mask']
    match_mask = result['match_mask']
    
    rows, cols = grid_points.shape[:2]
    unmatched_count = 0
    for row in range(rows):
        for col in range(cols):
            pt = tuple(grid_points[row, col].astype(int))
            if match_mask[row, col]:
                pass  # 匹配的点会用黄色覆盖
            elif valid_mask[row, col]:
                # 蓝色十字：预期有圆点但未匹配
                cv2.drawMarker(vis, pt, (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
                unmatched_count += 1
    
    # 绘制匹配的角点（黄色大圆）
    for corner in result['matched_corners']:
        pt = tuple(corner[0].astype(int))
        cv2.circle(vis, pt, 10, (0, 255, 255), -1)
    
    # 绘制AprilTag
    tag_corners = result['apriltag_info']['corners'].astype(int)
    cv2.polylines(vis, [tag_corners], True, (0, 255, 0), 3)
    
    # 统计未匹配的blob
    matched_blob_set = set()
    for idx in result['matched_indices']:
        row, col = idx
        grid_pos = grid_points[row, col]
        for i, blob_pos in enumerate(result['blob_points']):
            if np.linalg.norm(blob_pos - grid_pos) < result.get('max_distance', 25.0):
                matched_blob_set.add(i)
                break
    
    unmatched_blobs = len(result['blob_points']) - len(matched_blob_set)
    
    # 添加信息
    info = [
        f"Config: {config_name}",
        f"Threshold: {result.get('max_distance', 25.0):.1f}px",
        f"Matched: {result['match_count']}/{result['valid_count']}",
        f"Rate: {result['match_rate']:.1f}%",
        f"Unmatched blobs: {unmatched_blobs}",
        f"Unmatched grids: {unmatched_count}"
    ]
    
    y = 30
    for text in info:
        # 半透明背景
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        overlay = vis.copy()
        cv2.rectangle(overlay, (5, y-22), (15 + text_size[0], y+5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)
        y += 30
    
    # 添加图例
    legend_y = vis.shape[0] - 120
    legend_items = [
        ((0, 255, 0), "Green: Detected blob"),
        ((0, 255, 255), "Yellow: Matched point"),
        ((255, 0, 0), "Blue cross: Expected but unmatched"),
    ]
    
    for i, (color, text) in enumerate(legend_items):
        y = legend_y + i * 50
        if i == 0:
            cv2.circle(vis, (20, y), 6, color, 2)
        elif i == 1:
            cv2.circle(vis, (20, y), 8, color, -1)
        else:
            cv2.drawMarker(vis, (20, y), color, cv2.MARKER_CROSS, 10, 2)
        
        cv2.putText(vis, text, (40, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)
    
    return vis


# 主程序
if __name__ == '__main__':
    configs = [
        {
            'name': 'A2-优化: 圆65mm, Tag71mm, 内部, 大阈值',
            'circle_spacing': 0.065,
            'apriltag_size': 0.0714,
            'apriltag_position': 'right_top_inside',
            'max_distance': 200.0  # 增大阈值
        },
        {
            'name': 'A2-原始: 圆65mm, Tag71mm, 内部',
            'circle_spacing': 0.065,
            'apriltag_size': 0.0714,
            'apriltag_position': 'right_top_inside',
            'max_distance': 200.0
        },
        {
            'name': 'B2-优化: 圆6.5mm, Tag7.1mm, 内部, 大阈值',
            'circle_spacing': 0.0065,
            'apriltag_size': 0.0071,
            'apriltag_position': 'right_top_inside',
            'max_distance': 200.0
        },
        {
            'name': 'B2-原始: 圆6.5mm, Tag7.1mm, 内部',
            'circle_spacing': 0.0065,
            'apriltag_size': 0.0071,
            'apriltag_position': 'right_top_inside',
            'max_distance': 200.0
        },
    ]
    
    image = cv2.imread('data/1764744101_27_picture.png')
    
    if image is None:
        print("无法读取图像")
        exit(1)
    
    print("="*80)
    print("测试所有配置")
    print("="*80)
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}")
        result = test_config(image, config)
        
        if result:
            print(f"  ✅ {result['match_count']}/{result['valid_count']} ({result['match_rate']:.1f}%)")
            results.append({'config': config, 'result': result})
        else:
            print(f"  ❌ 失败")
    
    if results:
        print("\n" + "="*80)
        print("结果排名")
        print("="*80)
        
        results.sort(key=lambda x: x['result']['match_rate'], reverse=True)
        
        for i, r in enumerate(results):
            print(f"{i+1}. {r['config']['name']}: {r['result']['match_rate']:.1f}%")
        
        best = results[0]
        print(f"\n最佳: {best['config']['name']}")
        
        vis = visualize_result(image, best['result'], best['config']['name'])
        cv2.imwrite('best_result.png', vis)
        cv2.imshow('Best Result', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n❌ 所有配置都失败了")
