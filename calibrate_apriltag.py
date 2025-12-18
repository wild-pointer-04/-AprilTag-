#!/usr/bin/env python3
"""
AprilTag 参数校准工具

帮助你找出正确的参数配置
"""

import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from pupil_apriltags import Detector
    USING_PUPIL = True
except ImportError:
    import apriltag
    USING_PUPIL = False


def build_blob_detector():
    """构建blob检测器"""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.6
    return cv2.SimpleBlobDetector_create(params)


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
    
    return {
        'corners': corners,
        'center': center,
        'tag_id': tag_id
    }


def calibrate_parameters_interactive(image_path):
    """
    交互式参数校准
    
    步骤：
    1. 检测 AprilTag 和所有 blob
    2. 手动测量实际间距
    3. 自动计算正确的参数
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("="*80)
    print("AprilTag 参数校准工具")
    print("="*80)
    
    # 1. 检测 AprilTag
    print("\n步骤 1: 检测 AprilTag...")
    apriltag_info = detect_apriltag(gray)
    
    if apriltag_info is None:
        print("❌ 未检测到 AprilTag！")
        return
    
    print(f"✅ 检测到 AprilTag ID: {apriltag_info['tag_id']}")
    print(f"   中心位置: ({apriltag_info['center'][0]:.1f}, {apriltag_info['center'][1]:.1f})")
    
    # 计算 AprilTag 像素尺寸
    corners = apriltag_info['corners']
    tag_width_px = np.linalg.norm(corners[1] - corners[0])
    tag_height_px = np.linalg.norm(corners[3] - corners[0])
    tag_size_px = (tag_width_px + tag_height_px) / 2.0
    
    print(f"   像素尺寸: {tag_size_px:.2f} px")
    
    # 2. 检测所有 blob
    print("\n步骤 2: 检测所有圆点...")
    blob_detector = build_blob_detector()
    keypoints = blob_detector.detect(gray)
    
    if len(keypoints) == 0:
        print("❌ 未检测到任何圆点！")
        return
    
    blob_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    print(f"✅ 检测到 {len(blob_points)} 个圆点")
    
    # 3. 找到距离 AprilTag 最近的几个圆点
    print("\n步骤 3: 分析 AprilTag 周围的圆点...")
    tag_center = apriltag_info['center']
    
    distances = np.linalg.norm(blob_points - tag_center, axis=1)
    nearest_indices = np.argsort(distances)[:5]
    
    print("\n最近的5个圆点：")
    for i, idx in enumerate(nearest_indices):
        dist = distances[idx]
        blob_pos = blob_points[idx]
        print(f"  {i+1}. 圆点 @ ({blob_pos[0]:.1f}, {blob_pos[1]:.1f}), 距离={dist:.1f}px")
    
    # 4. 计算相邻圆点的间距
    print("\n步骤 4: 估算圆点间距...")
    
    # 使用所有圆点对的距离直方图来估算间距
    all_distances = []
    for i in range(min(50, len(blob_points))):  # 只用前50个点加速
        for j in range(i+1, min(50, len(blob_points))):
            dist = np.linalg.norm(blob_points[i] - blob_points[j])
            if 20 < dist < 200:  # 过滤明显不是相邻的点
                all_distances.append(dist)
    
    if len(all_distances) > 0:
        all_distances = np.array(all_distances)
        hist, bins = np.histogram(all_distances, bins=50)
        peak_idx = np.argmax(hist)
        estimated_spacing_px = (bins[peak_idx] + bins[peak_idx + 1]) / 2.0
        
        print(f"✅ 估算的圆点间距: {estimated_spacing_px:.2f} px")
    else:
        print("❌ 无法估算圆点间距")
        estimated_spacing_px = None
    
    # 5. 可视化
    print("\n步骤 5: 生成可视化...")
    vis = image.copy()
    
    # 绘制所有 blob
    for kp in keypoints:
        pt = (int(kp.pt[0]), int(kp.pt[1]))
        cv2.circle(vis, pt, int(kp.size/2), (0, 255, 0), 2)
    
    # 绘制 AprilTag
    tag_corners = corners.astype(int)
    cv2.polylines(vis, [tag_corners], True, (0, 255, 0), 3)
    cv2.drawMarker(vis, tuple(tag_center.astype(int)), (0, 255, 0),
                  cv2.MARKER_DIAMOND, 20, 3)
    
    # 绘制最近的圆点
    for idx in nearest_indices:
        pt = tuple(blob_points[idx].astype(int))
        cv2.circle(vis, pt, 12, (0, 0, 255), 3)
        cv2.line(vis, tuple(tag_center.astype(int)), pt, (255, 0, 0), 2)
    
    # 添加信息
    info_text = [
        f"AprilTag size: {tag_size_px:.1f} px",
        f"Blobs detected: {len(blob_points)}",
        f"Estimated spacing: {estimated_spacing_px:.1f} px" if estimated_spacing_px else "N/A",
    ]
    
    y = 30
    for text in info_text:
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        y += 35
    
    cv2.imshow('Parameter Calibration', vis)
    print("\n按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 6. 计算建议的参数
    print("\n" + "="*80)
    print("参数建议")
    print("="*80)
    
    print("\n请回答以下问题以计算正确的参数：")
    print("-"*80)
    
    # 获取 AprilTag 实际尺寸
    print("\n1. AprilTag 的实际尺寸是多少？")
    print("   (黑色正方形的外边长，单位：毫米或米)")
    apriltag_real_size_input = input("   输入（例如：7.1 或 0.0071）: ").strip()
    
    try:
        apriltag_real_size = float(apriltag_real_size_input)
        
        # 自动判断单位
        if apriltag_real_size > 1:
            apriltag_real_size_m = apriltag_real_size / 1000.0  # 毫米转米
            print(f"   解释为: {apriltag_real_size} mm = {apriltag_real_size_m} m")
        else:
            apriltag_real_size_m = apriltag_real_size
            print(f"   解释为: {apriltag_real_size_m} m = {apriltag_real_size_m * 1000} mm")
    except ValueError:
        print("   输入无效，使用默认值 7.1mm")
        apriltag_real_size_m = 0.0071
    
    # 获取圆点间距
    print("\n2. 相邻圆点的实际间距是多少？")
    print("   (圆心到圆心的距离，单位：毫米或米)")
    circle_spacing_input = input("   输入（例如：65 或 0.065）: ").strip()
    
    try:
        circle_spacing = float(circle_spacing_input)
        
        # 自动判断单位
        if circle_spacing > 1:
            circle_spacing_m = circle_spacing / 1000.0
            print(f"   解释为: {circle_spacing} mm = {circle_spacing_m} m")
        else:
            circle_spacing_m = circle_spacing
            print(f"   解释为: {circle_spacing_m} m = {circle_spacing_m * 1000} mm")
    except ValueError:
        print("   输入无效，使用默认值 65mm")
        circle_spacing_m = 0.065
    
    # 获取 AprilTag 相对位置
    print("\n3. AprilTag 相对于网格的位置是？")
    print("   a) 在右上角，替代了第15列第1行的圆点")
    print("   b) 在右上角外部，贴在板子边缘")
    print("   c) 在其他位置")
    position_input = input("   选择 (a/b/c): ").strip().lower()
    
    if position_input == 'a':
        apriltag_position = 'right_top_inside'
        print("   选择: 内部替代圆点")
    elif position_input == 'b':
        apriltag_position = 'right_top'
        print("   选择: 外部边缘")
    else:
        apriltag_position = 'right_top'
        print("   默认: 外部边缘")
    
    # 7. 计算和验证
    print("\n" + "="*80)
    print("计算结果")
    print("="*80)
    
    # 计算像素/米比例
    pixel_per_meter = tag_size_px / apriltag_real_size_m
    
    print(f"\n从 AprilTag 计算:")
    print(f"  像素/米比例: {pixel_per_meter:.2f} px/m")
    print(f"  圆点间距应该是: {circle_spacing_m * pixel_per_meter:.2f} px")
    
    if estimated_spacing_px:
        print(f"\n从圆点分布估算:")
        print(f"  实测圆点间距: {estimated_spacing_px:.2f} px")
        
        # 检查一致性
        expected_spacing = circle_spacing_m * pixel_per_meter
        error = abs(estimated_spacing_px - expected_spacing)
        error_pct = error / expected_spacing * 100
        
        print(f"\n一致性检查:")
        print(f"  预期间距: {expected_spacing:.2f} px")
        print(f"  实测间距: {estimated_spacing_px:.2f} px")
        print(f"  误差: {error:.2f} px ({error_pct:.1f}%)")
        
        if error_pct < 10:
            print(f"  ✅ 误差 < 10%，参数可能正确")
        else:
            print(f"  ⚠️ 误差 >= 10%，参数可能有误")
            print(f"\n  可能的原因:")
            print(f"    - AprilTag 实际尺寸不是 {apriltag_real_size_m*1000:.1f}mm")
            print(f"    - 圆点间距不是 {circle_spacing_m*1000:.1f}mm")
            print(f"    - 图像有畸变未矫正")
    
    # 8. 输出推荐配置
    print("\n" + "="*80)
    print("推荐的代码配置")
    print("="*80)
    
    print("\ndetector = AprilTagGuidedGridDetector(")
    print(f"    pattern_size=(15, 15),")
    print(f"    circle_spacing={circle_spacing_m},  # {circle_spacing_m*1000:.1f}mm")
    print(f"    apriltag_size={apriltag_real_size_m},  # {apriltag_real_size_m*1000:.1f}mm")
    print(f"    max_match_distance={estimated_spacing_px/2 if estimated_spacing_px else 25.0:.1f},")
    print(f"    apriltag_position='{apriltag_position}'")
    print(")")
    
    print("\n" + "="*80)
    
    # 9. 测试推算
    print("\n是否要测试网格推算？(y/n): ", end="")
    test_input = input().strip().lower()
    
    if test_input == 'y':
        test_grid_estimation(
            image, apriltag_info, blob_points, keypoints,
            circle_spacing_m, apriltag_real_size_m, apriltag_position
        )


def test_grid_estimation(image, apriltag_info, blob_points, keypoints,
                        circle_spacing, apriltag_size, apriltag_position):
    """测试网格推算是否正确"""
    
    print("\n测试网格推算...")
    
    tag_center = apriltag_info['center']
    tag_corners = apriltag_info['corners']
    
    # 计算方向向量
    top_left = tag_corners[0]
    top_right = tag_corners[1]
    bottom_left = tag_corners[3]
    
    tag_x_vec = top_right - top_left
    tag_y_vec = bottom_left - top_left
    
    tag_x_len = np.linalg.norm(tag_x_vec)
    tag_y_len = np.linalg.norm(tag_y_vec)
    
    unit_x = tag_x_vec / tag_x_len
    unit_y = tag_y_vec / tag_y_len
    
    pixel_per_meter = (tag_x_len + tag_y_len) / (2.0 * apriltag_size)
    circle_spacing_px = circle_spacing * pixel_per_meter
    
    print(f"  圆点间距（像素）: {circle_spacing_px:.2f} px")
    
    # 确定 AprilTag 在网格中的偏移
    if apriltag_position == 'right_top_inside':
        offset_x = 14
        offset_y = 0
    else:
        offset_x = 14.5
        offset_y = 0
    
    # 计算网格原点
    grid_origin = tag_center - unit_x * offset_x * circle_spacing_px - unit_y * offset_y * circle_spacing_px
    
    print(f"  网格原点: ({grid_origin[0]:.1f}, {grid_origin[1]:.1f})")
    
    # 生成一些测试点
    test_positions = [
        (0, 0, "左上角"),
        (14, 0, "右上角"),
        (7, 7, "中心"),
        (0, 14, "左下角"),
        (14, 14, "右下角")
    ]
    
    vis = image.copy()
    
    # 绘制所有 blob（绿色）
    for kp in keypoints:
        pt = (int(kp.pt[0]), int(kp.pt[1]))
        cv2.circle(vis, pt, int(kp.size/2), (0, 255, 0), 2)
    
    # 绘制 AprilTag
    cv2.polylines(vis, [tag_corners.astype(int)], True, (0, 255, 0), 3)
    
    print("\n  测试点:")
    for col, row, label in test_positions:
        # 计算预期位置
        expected_pos = grid_origin + unit_x * col * circle_spacing_px + unit_y * row * circle_spacing_px
        
        # 找最近的 blob
        distances = np.linalg.norm(blob_points - expected_pos, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        nearest_blob = blob_points[min_idx]
        
        # 绘制预期位置（蓝色十字）
        pt_expected = tuple(expected_pos.astype(int))
        cv2.drawMarker(vis, pt_expected, (255, 0, 0), cv2.MARKER_CROSS, 20, 3)
        
        # 绘制最近的 blob（红色圆）
        pt_blob = tuple(nearest_blob.astype(int))
        cv2.circle(vis, pt_blob, 12, (0, 0, 255), 3)
        
        # 连线
        cv2.line(vis, pt_expected, pt_blob, (255, 255, 0), 2)
        
        # 标注
        cv2.putText(vis, f"({col},{row})", pt_expected,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        print(f"    {label} ({col},{row}): 预期=({expected_pos[0]:.1f},{expected_pos[1]:.1f}), "
              f"最近blob距离={min_dist:.1f}px")
    
    cv2.imshow('Grid Estimation Test', vis)
    print("\n蓝色十字 = 预期位置, 红色圆 = 最近的blob")
    print("按任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python calibrate_apriltag.py <图像路径>")
        print("示例: python calibrate_apriltag.py data/1764744104_27_picture.png")
    else:
        calibrate_parameters_interactive(sys.argv[1])