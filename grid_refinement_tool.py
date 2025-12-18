#!/usr/bin/env python3
"""
网格精细校准工具

策略：
1. 用 AprilTag 得到初步的网格位置和方向
2. 用实际检测到的 blob 来微调网格的原点和间距
"""

import cv2
import numpy as np
from standalone_quick_test import (
    detect_apriltag, detect_blobs, estimate_grid,
    match_blobs_to_grid, simple_greedy_matching
)


def refine_grid_with_blobs(grid_points, valid_mask, blob_points, 
                           unit_x, unit_y, circle_spacing_px):
    """
    使用实际检测到的 blob 来微调网格参数
    
    策略：
    1. 先用初步网格做粗匹配
    2. 计算匹配点的平均偏差
    3. 用偏差来校正网格原点
    4. 重新匹配
    """
    
    print("\n[网格精细校准]")
    
    # 1. 粗匹配
    _, matched_indices, _ = match_blobs_to_grid(
        blob_points, grid_points, valid_mask, max_distance=60.0
    )
    
    if matched_indices is None or len(matched_indices) < 10:
        print("  初步匹配点太少，无法校准")
        return grid_points, circle_spacing_px, unit_x, unit_y
    
    print(f"  初步匹配: {len(matched_indices)} 个点")
    
    # 2. 计算每个匹配点的偏差
    offsets = []
    for idx, (row, col) in enumerate(matched_indices):
        expected = grid_points[row, col]
        actual = blob_points[idx]
        offset = actual - expected
        offsets.append(offset)
    
    offsets = np.array(offsets)
    
    # 3. 计算平均偏差（这是网格原点的系统性误差）
    mean_offset = np.mean(offsets, axis=0)
    std_offset = np.std(offsets, axis=0)
    
    print(f"  平均偏差: ({mean_offset[0]:.2f}, {mean_offset[1]:.2f}) px")
    print(f"  标准差: ({std_offset[0]:.2f}, {std_offset[1]:.2f}) px")
    
    # 4. 校正网格原点
    # 如果平均偏差超过5px，就校正
    if np.linalg.norm(mean_offset) > 5.0:
        print(f"  应用偏差校正...")
        
        # 更新整个网格
        refined_grid = grid_points.copy()
        for row in range(grid_points.shape[0]):
            for col in range(grid_points.shape[1]):
                refined_grid[row, col] = grid_points[row, col] + mean_offset
        
        # 5. 可选：用最小二乘法优化间距和方向
        # 这里简化处理，只校正原点
        
        return refined_grid, circle_spacing_px, unit_x, unit_y
    else:
        print(f"  偏差较小，无需校正")
        return grid_points, circle_spacing_px, unit_x, unit_y


def detect_with_refinement(image, circle_spacing, apriltag_size, 
                          apriltag_position='right_top_inside',
                          max_distance=60.0):
    """
    带精细校准的检测
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 检测 AprilTag 和 blob
    apriltag_info = detect_apriltag(gray)
    if apriltag_info is None:
        return None
    
    blob_points, keypoints = detect_blobs(gray)
    if len(blob_points) == 0:
        return None
    
    print(f"\n检测到 {len(blob_points)} 个 blob")
    
    # 2. 初步网格估算
    grid_points, valid_mask = estimate_grid(
        apriltag_info,
        circle_spacing,
        apriltag_size,
        apriltag_position,
        image.shape
    )
    
    # 3. 计算方向向量（需要从 estimate_grid 中提取）
    tag_corners = apriltag_info['corners']
    
    # 重新识别角点
    y_coords = [corner[1] for corner in tag_corners]
    corners_with_idx = [(i, tag_corners[i]) for i in range(4)]
    corners_with_idx.sort(key=lambda x: x[1][1])
    
    top_two = corners_with_idx[:2]
    bottom_two = corners_with_idx[2:]
    
    top_two.sort(key=lambda x: x[1][0])
    bottom_two.sort(key=lambda x: x[1][0])
    
    top_left = top_two[0][1]
    top_right = top_two[1][1]
    bottom_left = bottom_two[0][1]
    
    tag_x_vec = top_right - top_left
    tag_y_vec = bottom_left - top_left
    
    tag_x_len = np.linalg.norm(tag_x_vec)
    tag_y_len = np.linalg.norm(tag_y_vec)
    
    unit_x = tag_x_vec / tag_x_len
    unit_y = tag_y_vec / tag_y_len
    
    pixel_per_meter = (tag_x_len + tag_y_len) / (2.0 * apriltag_size)
    circle_spacing_px = circle_spacing * pixel_per_meter
    
    # 4. 精细校准
    refined_grid, refined_spacing, refined_x, refined_y = refine_grid_with_blobs(
        grid_points, valid_mask, blob_points, unit_x, unit_y, circle_spacing_px
    )
    
    # 5. 用精细校准后的网格重新匹配
    matched_corners, matched_indices, match_mask = match_blobs_to_grid(
        blob_points, refined_grid, valid_mask, max_distance=max_distance
    )
    
    if matched_corners is None:
        return None
    
    valid_count = np.sum(valid_mask)
    match_count = len(matched_corners)
    
    print(f"\n精细校准后匹配: {match_count}/{valid_count} ({match_count/valid_count*100:.1f}%)")
    
    return {
        'success': True,
        'match_count': match_count,
        'valid_count': valid_count,
        'match_rate': match_count / valid_count * 100,
        'matched_corners': matched_corners,
        'matched_indices': matched_indices,
        'grid_points': refined_grid,
        'valid_mask': valid_mask,
        'match_mask': match_mask,
        'blob_points': blob_points,
        'keypoints': keypoints,
        'apriltag_info': apriltag_info,
        'max_distance': max_distance
    }


def visualize_comparison(image, result_before, result_after):
    """对比显示精细校准前后的效果"""
    
    # 创建并排显示
    h, w = image.shape[:2]
    vis = np.zeros((h, w*2, 3), dtype=np.uint8)
    
    # 左边：校准前
    vis_before = image.copy()
    
    # 绘制blob
    for kp in result_before['keypoints']:
        pt = (int(kp.pt[0]), int(kp.pt[1]))
        cv2.circle(vis_before, pt, int(kp.size/2), (0, 255, 0), 2)
    
    # 绘制未匹配的网格点（蓝色十字）
    grid = result_before['grid_points']
    valid = result_before['valid_mask']
    match = result_before['match_mask']
    
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if valid[row, col] and not match[row, col]:
                pt = tuple(grid[row, col].astype(int))
                cv2.drawMarker(vis_before, pt, (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
    
    # 绘制匹配点（黄色）
    for corner in result_before['matched_corners']:
        pt = tuple(corner[0].astype(int))
        cv2.circle(vis_before, pt, 10, (0, 255, 255), -1)
    
    cv2.putText(vis_before, "Before Refinement", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_before, f"{result_before['match_rate']:.1f}%", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # 右边：校准后
    vis_after = image.copy()
    
    for kp in result_after['keypoints']:
        pt = (int(kp.pt[0]), int(kp.pt[1]))
        cv2.circle(vis_after, pt, int(kp.size/2), (0, 255, 0), 2)
    
    grid = result_after['grid_points']
    valid = result_after['valid_mask']
    match = result_after['match_mask']
    
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if valid[row, col] and not match[row, col]:
                pt = tuple(grid[row, col].astype(int))
                cv2.drawMarker(vis_after, pt, (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
    
    for corner in result_after['matched_corners']:
        pt = tuple(corner[0].astype(int))
        cv2.circle(vis_after, pt, 10, (0, 255, 255), -1)
    
    cv2.putText(vis_after, "After Refinement", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_after, f"{result_after['match_rate']:.1f}%", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # 改进增量
    improvement = result_after['match_rate'] - result_before['match_rate']
    cv2.putText(vis_after, f"+{improvement:.1f}%", (10, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    vis[:, :w] = vis_before
    vis[:, w:] = vis_after
    
    return vis


if __name__ == '__main__':
    image = cv2.imread('data/1764744101_27_picture.png')
    
    if image is None:
        print("无法读取图像")
        exit(1)
    
    print("="*80)
    print("网格精细校准测试")
    print("="*80)
    
    # 配置
    circle_spacing = 0.065  # 65mm
    apriltag_size = 0.0714  # 71.4mm
    
    # 1. 不精细校准（使用原始的 estimate_grid）
    print("\n[测试1: 不精细校准]")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    apriltag_info = detect_apriltag(gray)
    blob_points, keypoints = detect_blobs(gray)
    
    grid_points, valid_mask = estimate_grid(
        apriltag_info, circle_spacing, apriltag_size,
        'right_top_inside', image.shape
    )
    
    matched_corners, matched_indices, match_mask = match_blobs_to_grid(
        blob_points, grid_points, valid_mask, max_distance=60.0
    )
    
    result_before = {
        'matched_corners': matched_corners,
        'matched_indices': matched_indices,
        'grid_points': grid_points,
        'valid_mask': valid_mask,
        'match_mask': match_mask,
        'blob_points': blob_points,
        'keypoints': keypoints,
        'match_count': len(matched_corners),
        'valid_count': np.sum(valid_mask),
        'match_rate': len(matched_corners) / np.sum(valid_mask) * 100
    }
    
    print(f"匹配率: {result_before['match_rate']:.1f}%")
    
    # 2. 精细校准
    print("\n[测试2: 精细校准]")
    result_after = detect_with_refinement(
        image, circle_spacing, apriltag_size,
        'right_top_inside', max_distance=60.0
    )
    
    if result_after and result_after['success']:
        # 对比显示
        vis = visualize_comparison(image, result_before, result_after)
        
        cv2.imwrite('grid_refinement_comparison.png', vis)
        cv2.imshow('Before vs After Refinement', vis)
        
        print(f"\n改进: {result_after['match_rate'] - result_before['match_rate']:.1f}%")
        print("按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("精细校准失败")