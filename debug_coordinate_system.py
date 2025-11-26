#!/usr/bin/env python3
"""
调试坐标系建立问题
"""

import cv2
import numpy as np
import os
import sys

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.apriltag_coordinate_system import AprilTagCoordinateSystem
from src.detect_grid_improved import try_find_adaptive, refine
from src.utils import load_camera_intrinsics, get_camera_intrinsics


def debug_coordinate_system(image_path: str):
    """调试坐标系建立过程"""
    
    print(f"调试图像: {image_path}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"图像尺寸: {w} x {h}")
    
    # 使用默认相机内参
    K, dist = get_camera_intrinsics(h, w)
    
    # 畸变矫正
    undistorted = cv2.undistort(image, K, dist)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    
    # 检测圆点网格
    print("检测圆点网格...")
    ok, corners, blob_keypoints = try_find_adaptive(gray, 15, 15, symmetric=True)
    
    if not ok or corners is None:
        print("❌ 未检测到圆点网格")
        return
    
    # 精化角点
    corners_refined = refine(gray, corners)
    board_corners = corners_refined.reshape(-1, 2)
    print(f"✅ 检测到 {len(board_corners)} 个圆点")
    
    # 初始化AprilTag坐标系建立器 - 尝试不同的tag family
    tag_families = ['tag36h11', 'tag25h9', 'tag16h5', 'tagStandard41h12']
    coord_system = None
    detection = None
    
    for tag_family in tag_families:
        print(f"尝试AprilTag家族: {tag_family}")
        coord_system = AprilTagCoordinateSystem(
            tag_family=tag_family,
            tag_size=20.0,
            board_spacing=10.0
        )
        
        # 检测AprilTag
        detections = coord_system.detect_apriltag(gray)
        if len(detections) > 0:
            detection = detections[0]
            print(f"✅ 使用 {tag_family} 检测到AprilTag")
            break
        else:
            print(f"❌ {tag_family} 未检测到AprilTag")
    
    if detection is None:
        print("❌ 所有AprilTag家族都未检测到标签")
        return
    
    # AprilTag已经在上面检测过了
    tag_center = detection.center
    tag_corners = detection.corners
    
    print(f"AprilTag信息:")
    print(f"  ID: {detection.tag_id}")
    print(f"  中心: ({tag_center[0]:.1f}, {tag_center[1]:.1f})")
    print(f"  角点: {tag_corners}")
    
    # 找最近角点
    distances = np.linalg.norm(board_corners - tag_center, axis=1)
    nearest_idx = np.argmin(distances)
    nearest_corner = board_corners[nearest_idx]
    
    print(f"最近角点分析:")
    print(f"  索引: {nearest_idx}")
    print(f"  坐标: ({nearest_corner[0]:.1f}, {nearest_corner[1]:.1f})")
    print(f"  距离: {distances[nearest_idx]:.2f}px")
    
    # 创建可视化图像
    vis_image = undistorted.copy()
    
    # 绘制所有圆点
    for i, corner in enumerate(board_corners):
        color = (0, 255, 0) if i == nearest_idx else (255, 0, 0)
        cv2.circle(vis_image, tuple(corner.astype(int)), 3, color, -1)
        if i == nearest_idx:
            cv2.putText(vis_image, f"Origin({i})", 
                       (int(corner[0])+10, int(corner[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 绘制AprilTag
    tag_corners_int = tag_corners.astype(int)
    cv2.polylines(vis_image, [tag_corners_int], True, (0, 255, 255), 2)
    cv2.circle(vis_image, tuple(tag_center.astype(int)), 5, (0, 255, 255), -1)
    cv2.putText(vis_image, f"AprilTag({detection.tag_id})", 
               (int(tag_center[0])+10, int(tag_center[1])+10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 绘制连接线
    cv2.line(vis_image, tuple(tag_center.astype(int)), tuple(nearest_corner.astype(int)), 
             (255, 255, 0), 2)
    
    # 计算AprilTag方向
    tag_x_direction_2d = tag_corners[1] - tag_corners[0]  # 右下 - 左下
    tag_x_direction_2d = tag_x_direction_2d / np.linalg.norm(tag_x_direction_2d)
    
    tag_y_direction_2d = tag_corners[3] - tag_corners[0]  # 左上 - 左下
    tag_y_direction_2d = tag_y_direction_2d / np.linalg.norm(tag_y_direction_2d)
    
    print(f"AprilTag方向:")
    print(f"  X轴: ({tag_x_direction_2d[0]:.3f}, {tag_x_direction_2d[1]:.3f})")
    print(f"  Y轴: ({tag_y_direction_2d[0]:.3f}, {tag_y_direction_2d[1]:.3f})")
    
    # 绘制坐标轴（从最近角点开始）
    axis_length = 50.0
    origin = nearest_corner.astype(int)
    x_end = origin + (tag_x_direction_2d * axis_length).astype(int)
    y_end = origin + (tag_y_direction_2d * axis_length).astype(int)
    
    # X轴（红色）
    cv2.arrowedLine(vis_image, tuple(origin), tuple(x_end), (0, 0, 255), 3)
    cv2.putText(vis_image, "X", tuple(x_end + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Y轴（绿色）
    cv2.arrowedLine(vis_image, tuple(origin), tuple(y_end), (0, 255, 0), 3)
    cv2.putText(vis_image, "Y", tuple(y_end + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 添加信息文本
    info_text = [
        f"AprilTag ID: {detection.tag_id}",
        f"Tag Center: ({tag_center[0]:.1f}, {tag_center[1]:.1f})",
        f"Origin Index: {nearest_idx}",
        f"Origin: ({nearest_corner[0]:.1f}, {nearest_corner[1]:.1f})",
        f"Distance: {distances[nearest_idx]:.2f}px"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(vis_image, text, (10, 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, text, (10, 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 保存结果
    output_path = 'debug_coordinate_system_result.png'
    cv2.imwrite(output_path, vis_image)
    print(f"✅ 调试结果已保存到: {output_path}")
    
    return vis_image


if __name__ == '__main__':
    # 尝试所有图像文件，包括comprehensive_test_result
    import glob
    
    image_files = (glob.glob('data/*.png') + glob.glob('data/*.jpg') + 
                  glob.glob('*.png') + glob.glob('temp_images/*.png'))
    
    if image_files:
        for image_file in image_files:
            print(f"\n=== 尝试图像: {image_file} ===")
            try:
                debug_coordinate_system(image_file)
                break  # 找到一个有AprilTag的就停止
            except Exception as e:
                print(f"处理失败: {e}")
                continue
    else:
        print("❌ 未找到图像文件")