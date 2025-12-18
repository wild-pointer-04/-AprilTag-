#!/usr/bin/env python3
"""
诊断工具：分析为什么有些blob没有匹配到网格

这个工具会：
1. 显示每个未匹配blob的最近网格点和距离
2. 显示每个未匹配网格点的最近blob和距离
3. 帮助你调整参数
"""

import cv2
import numpy as np
from typing import Dict


def diagnose_matching(result: Dict, detector) -> None:
    """
    诊断匹配问题
    
    Args:
        result: 检测结果
        detector: 检测器实例
    """
    if not result['success']:
        print("检测失败，无法诊断")
        return
    
    blob_points = result['blob_points']
    grid_points = result['grid_points']
    valid_mask = result['valid_mask']
    match_mask = result['match_mask']
    matched_indices = result['matched_indices']
    
    print("\n" + "="*80)
    print("匹配诊断报告")
    print("="*80)
    
    # 1. 找出所有有效的网格点
    valid_grid_list = []
    for row in range(detector.grid_rows):
        for col in range(detector.grid_cols):
            if valid_mask[row, col]:
                valid_grid_list.append({
                    'pos': grid_points[row, col],
                    'row': row,
                    'col': col,
                    'matched': match_mask[row, col]
                })
    
    # 2. 分析未匹配的blob
    # 找出哪些blob被匹配了
    matched_blob_indices = set()
    for idx in matched_indices:
        row, col = idx
        # 找到这个网格点匹配的blob
        grid_pos = grid_points[row, col]
        min_dist = float('inf')
        min_blob_idx = -1
        for i, blob_pos in enumerate(blob_points):
            dist = np.linalg.norm(blob_pos - grid_pos)
            if dist < min_dist:
                min_dist = dist
                min_blob_idx = i
        if min_dist < detector.max_match_distance:
            matched_blob_indices.add(min_blob_idx)
    
    unmatched_blobs = []
    for i, blob_pos in enumerate(blob_points):
        if i not in matched_blob_indices:
            # 找最近的网格点
            min_dist = float('inf')
            nearest_grid = None
            for grid_info in valid_grid_list:
                dist = np.linalg.norm(blob_pos - grid_info['pos'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_grid = grid_info
            
            unmatched_blobs.append({
                'index': i,
                'pos': blob_pos,
                'nearest_grid': nearest_grid,
                'distance': min_dist
            })
    
    # 3. 分析未匹配的网格点
    unmatched_grids = []
    for grid_info in valid_grid_list:
        if not grid_info['matched']:
            # 找最近的blob
            min_dist = float('inf')
            nearest_blob_idx = -1
            for i, blob_pos in enumerate(blob_points):
                dist = np.linalg.norm(blob_pos - grid_info['pos'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_blob_idx = i
            
            unmatched_grids.append({
                'row': grid_info['row'],
                'col': grid_info['col'],
                'pos': grid_info['pos'],
                'nearest_blob_idx': nearest_blob_idx,
                'distance': min_dist
            })
    
    # 4. 打印报告
    print(f"\n总统计：")
    print(f"  有效网格点数: {len(valid_grid_list)}")
    print(f"  检测到的blob数: {len(blob_points)}")
    print(f"  成功匹配数: {len(matched_indices)}")
    print(f"  未匹配的blob: {len(unmatched_blobs)}")
    print(f"  未匹配的网格点: {len(unmatched_grids)}")
    print(f"  当前匹配距离阈值: {detector.max_match_distance:.1f} px")
    
    # 5. 详细分析未匹配的blob
    if len(unmatched_blobs) > 0:
        print(f"\n未匹配的Blob详情（共{len(unmatched_blobs)}个）：")
        print("-" * 80)
        
        # 按距离排序
        unmatched_blobs.sort(key=lambda x: x['distance'])
        
        for i, info in enumerate(unmatched_blobs[:10]):  # 只显示前10个
            nearest = info['nearest_grid']
            print(f"{i+1}. Blob #{info['index']} @ ({info['pos'][0]:.1f}, {info['pos'][1]:.1f})")
            if nearest:
                print(f"   最近网格点: ({nearest['row']}, {nearest['col']}) "
                      f"@ ({nearest['pos'][0]:.1f}, {nearest['pos'][1]:.1f})")
                print(f"   距离: {info['distance']:.2f} px", end="")
                
                if info['distance'] > detector.max_match_distance:
                    print(f" ❌ 超过阈值 {detector.max_match_distance:.1f} px")
                elif nearest['matched']:
                    print(f" ⚠️  该网格点已被其他blob匹配")
                else:
                    print(f" ❓ 距离在阈值内但未匹配（可能是匈牙利算法的次优解）")
            print()
        
        if len(unmatched_blobs) > 10:
            print(f"... 还有 {len(unmatched_blobs) - 10} 个未匹配的blob\n")
    
    # 6. 详细分析未匹配的网格点
    if len(unmatched_grids) > 0:
        print(f"\n未匹配的网格点详情（共{len(unmatched_grids)}个）：")
        print("-" * 80)
        
        # 按距离排序
        unmatched_grids.sort(key=lambda x: x['distance'])
        
        for i, info in enumerate(unmatched_grids[:10]):  # 只显示前10个
            print(f"{i+1}. 网格点 ({info['row']}, {info['col']}) "
                  f"@ ({info['pos'][0]:.1f}, {info['pos'][1]:.1f})")
            print(f"   最近blob: #{info['nearest_blob_idx']} "
                  f"@ ({blob_points[info['nearest_blob_idx']][0]:.1f}, "
                  f"{blob_points[info['nearest_blob_idx']][1]:.1f})")
            print(f"   距离: {info['distance']:.2f} px", end="")
            
            if info['distance'] > detector.max_match_distance:
                print(f" ❌ 超过阈值 {detector.max_match_distance:.1f} px")
            elif info['nearest_blob_idx'] in matched_blob_indices:
                print(f" ⚠️  该blob已被其他网格点匹配")
            else:
                print(f" ❓ 距离在阈值内但未匹配（可能是匈牙利算法的次优解）")
            print()
        
        if len(unmatched_grids) > 10:
            print(f"... 还有 {len(unmatched_grids) - 10} 个未匹配的网格点\n")
    
    # 7. 给出建议
    print("\n" + "="*80)
    print("诊断建议：")
    print("="*80)
    
    # 计算需要的阈值
    all_distances = [info['distance'] for info in unmatched_blobs + unmatched_grids]
    if all_distances:
        max_unmatch_dist = max(all_distances)
        avg_unmatch_dist = np.mean(all_distances)
        
        print(f"\n1. 匹配距离分析：")
        print(f"   未匹配点的最大距离: {max_unmatch_dist:.2f} px")
        print(f"   未匹配点的平均距离: {avg_unmatch_dist:.2f} px")
        print(f"   当前阈值: {detector.max_match_distance:.1f} px")
        
        if max_unmatch_dist > detector.max_match_distance:
            suggested_threshold = max_unmatch_dist * 1.1
            print(f"\n   💡 建议增大阈值到: {suggested_threshold:.1f} px")
    
    # 检查网格位置是否准确
    if len(unmatched_grids) > 0:
        distances_in_threshold = [info['distance'] for info in unmatched_grids 
                                 if info['distance'] < detector.max_match_distance]
        if len(distances_in_threshold) > len(unmatched_grids) * 0.3:
            print(f"\n2. ⚠️  警告：有{len(distances_in_threshold)}个网格点在阈值内但未匹配")
            print(f"   这可能是因为：")
            print(f"   - AprilTag位置参数不准确")
            print(f"   - 圆点间距参数不准确")
            print(f"   - AprilTag尺寸参数不准确")
            print(f"\n   💡 建议：检查以下参数是否正确")
            print(f"      - circle_spacing: {detector.circle_spacing}m")
            print(f"      - apriltag_size: {detector.apriltag_size}m")
            print(f"      - apriltag_position: {detector.apriltag_position}")
    
    print("\n" + "="*80 + "\n")


def visualize_unmatched_details(image: np.ndarray, result: Dict, detector) -> np.ndarray:
    """
    创建详细的未匹配可视化，用线连接未匹配的blob和它们最近的网格点
    """
    vis = image.copy()
    
    if not result['success']:
        return vis
    
    blob_points = result['blob_points']
    grid_points = result['grid_points']
    valid_mask = result['valid_mask']
    match_mask = result['match_mask']
    matched_indices = result['matched_indices']
    
    # 找出匹配的blob
    matched_blob_set = set()
    for idx in matched_indices:
        row, col = idx
        grid_pos = grid_points[row, col]
        for i, blob_pos in enumerate(blob_points):
            if np.linalg.norm(blob_pos - grid_pos) < detector.max_match_distance:
                matched_blob_set.add(i)
                break
    
    # 绘制未匹配的blob及其到最近网格点的连线
    for i, blob_pos in enumerate(blob_points):
        if i not in matched_blob_set:
            # 找最近的未匹配网格点
            min_dist = float('inf')
            nearest_grid_pos = None
            
            for row in range(detector.grid_rows):
                for col in range(detector.grid_cols):
                    if valid_mask[row, col] and not match_mask[row, col]:
                        dist = np.linalg.norm(blob_pos - grid_points[row, col])
                        if dist < min_dist:
                            min_dist = dist
                            nearest_grid_pos = grid_points[row, col]
            
            if nearest_grid_pos is not None:
                # 绘制连线
                pt1 = tuple(blob_pos.astype(int))
                pt2 = tuple(nearest_grid_pos.astype(int))
                
                # 根据距离选择颜色
                if min_dist < detector.max_match_distance:
                    color = (0, 165, 255)  # 橙色：在阈值内但未匹配
                    thickness = 2
                else:
                    color = (0, 0, 255)    # 红色：超出阈值
                    thickness = 1
                
                cv2.line(vis, pt1, pt2, color, thickness)
                
                # 标注距离
                mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                cv2.putText(vis, f"{min_dist:.1f}", mid_pt,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis


# 使用示例
if __name__ == '__main__':
    # 在你的主代码中添加：
    
    # result = detector.detect(image)
    
    # 运行诊断
    # diagnose_matching(result, detector)
    
    # 生成详细可视化
    # vis_details = visualize_unmatched_details(image, result, detector)
    # cv2.imshow('Unmatched Details', vis_details)
    # cv2.waitKey(0)
    
    pass