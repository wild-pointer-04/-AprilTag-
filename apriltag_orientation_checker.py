#!/usr/bin/env python3
"""
AprilTag方向检查工具
帮助确认AprilTag的角点顺序和方向
"""

import cv2
import numpy as np

try:
    from pupil_apriltags import Detector
    USING_PUPIL = True
except ImportError:
    import apriltag
    USING_PUPIL = False


def check_apriltag_orientation(image_path):
    """检查AprilTag的方向"""
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测AprilTag
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
    
    detections = detector.detect(gray)
    
    if len(detections) == 0:
        print("未检测到AprilTag")
        return
    
    detection = detections[0]
    
    if USING_PUPIL:
        corners = np.array(detection.corners, dtype=np.float64)
        center = np.array(detection.center, dtype=np.float64)
        tag_id = detection.tag_id
    else:
        corners = np.array(detection.corners, dtype=np.float64)
        center = np.array(detection.center, dtype=np.float64)
        tag_id = detection.tag_id
    
    print("="*80)
    print("AprilTag 方向分析")
    print("="*80)
    
    print(f"\nAprilTag ID: {tag_id}")
    print(f"中心位置: ({center[0]:.1f}, {center[1]:.1f})")
    
    print("\n角点位置:")
    corner_names = ["0-左上(?)", "1-右上(?)", "2-右下(?)", "3-左下(?)"]
    for i, (corner, name) in enumerate(zip(corners, corner_names)):
        print(f"  角点{i} ({name}): ({corner[0]:.1f}, {corner[1]:.1f})")
    
    # 可视化
    vis = image.copy()
    
    # 绘制AprilTag外框
    cv2.polylines(vis, [corners.astype(int)], True, (0, 255, 0), 3)
    
    # 标记中心
    cv2.drawMarker(vis, tuple(center.astype(int)), (0, 255, 0),
                  cv2.MARKER_DIAMOND, 20, 3)
    cv2.putText(vis, "Center", tuple((center + 15).astype(int)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 标记每个角点
    colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    labels = ["0", "1", "2", "3"]
    
    for i, (corner, color, label) in enumerate(zip(corners, colors, labels)):
        pt = tuple(corner.astype(int))
        cv2.circle(vis, pt, 12, color, -1)
        cv2.putText(vis, label, (pt[0] + 15, pt[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    # 绘制方向向量
    # X方向：从角点0到角点1
    x_start = corners[0].astype(int)
    x_end = corners[1].astype(int)
    cv2.arrowedLine(vis, tuple(x_start), tuple(x_end), (0, 0, 255), 3, tipLength=0.2)
    mid_x = ((x_start + x_end) / 2).astype(int)
    cv2.putText(vis, "X-axis", tuple(mid_x + np.array([0, -20])),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Y方向：从角点0到角点3
    y_start = corners[0].astype(int)
    y_end = corners[3].astype(int)
    cv2.arrowedLine(vis, tuple(y_start), tuple(y_end), (255, 0, 0), 3, tipLength=0.2)
    mid_y = ((y_start + y_end) / 2).astype(int)
    cv2.putText(vis, "Y-axis", tuple(mid_y + np.array([20, 0])),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # 分析方向
    print("\n方向分析:")
    x_vec = corners[1] - corners[0]
    y_vec = corners[3] - corners[0]
    
    x_angle = np.degrees(np.arctan2(x_vec[1], x_vec[0]))
    y_angle = np.degrees(np.arctan2(y_vec[1], y_vec[0]))
    
    print(f"  X轴方向 (0→1): 角度={x_angle:.1f}°, 向量={x_vec}")
    print(f"  Y轴方向 (0→3): 角度={y_angle:.1f}°, 向量={y_vec}")
    
    # 判断AprilTag在图像中的位置
    image_center = np.array([image.shape[1]/2, image.shape[0]/2])
    tag_relative_pos = center - image_center
    
    h_pos = "右" if tag_relative_pos[0] > 0 else "左"
    v_pos = "下" if tag_relative_pos[1] > 0 else "上"
    
    print(f"\n相对图像中心: {h_pos}{v_pos}")
    
    # 检测blob找到最近的圆点
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
    
    blob_detector = cv2.SimpleBlobDetector_create(params)
    keypoints = blob_detector.detect(gray)
    blob_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    
    if len(blob_points) > 0:
        # 找最近的圆点
        distances = np.linalg.norm(blob_points - center, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_blob = blob_points[nearest_idx]
        nearest_dist = distances[nearest_idx]
        
        print(f"\n最近的圆点:")
        print(f"  位置: ({nearest_blob[0]:.1f}, {nearest_blob[1]:.1f})")
        print(f"  距离: {nearest_dist:.1f} px")
        
        # 在图上标记最近的圆点
        cv2.circle(vis, tuple(nearest_blob.astype(int)), 15, (0, 165, 255), 3)
        cv2.line(vis, tuple(center.astype(int)), tuple(nearest_blob.astype(int)),
                (0, 165, 255), 2)
        cv2.putText(vis, "Nearest blob", tuple((nearest_blob + 20).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # 分析最近圆点相对于AprilTag的方向
        blob_relative = nearest_blob - center
        blob_angle = np.degrees(np.arctan2(blob_relative[1], blob_relative[0]))
        
        print(f"  相对方向: 角度={blob_angle:.1f}°")
        
        if nearest_dist < 10:
            print("  ⚠️  AprilTag 可能 **替代** 了这个圆点的位置！")
        else:
            print("  ℹ️  AprilTag 在圆点 **外部**")
    
    # 添加图例
    legend_y = 30
    cv2.putText(vis, "Corner colors:", (image.shape[1] - 250, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    legend_y += 30
    for i, (color, label) in enumerate(zip(colors, labels)):
        cv2.circle(vis, (image.shape[1] - 230, legend_y), 8, color, -1)
        cv2.putText(vis, f"Corner {label}", (image.shape[1] - 210, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
    
    print("\n" + "="*80)
    
    cv2.imshow('AprilTag Orientation', vis)
    cv2.imwrite('apriltag_orientation.png', vis)
    print("\n按任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        image_path = 'data/1764744101_27_picture.png'
        print(f"使用默认图像: {image_path}")
    else:
        image_path = sys.argv[1]
    
    check_apriltag_orientation(image_path)
