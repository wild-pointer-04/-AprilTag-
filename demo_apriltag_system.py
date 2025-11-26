#!/usr/bin/env python3
"""
AprilTag坐标系建立演示脚本

这个脚本演示了如何使用AprilTag建立统一坐标系来解决相机倾斜检测中的坐标轴不确定性问题。
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
from src.utils import get_camera_intrinsics


def create_demo_image_with_apriltag():
    """
    创建一个包含AprilTag和圆点网格的演示图像
    
    注意：这只是一个概念演示，实际使用时需要真实的AprilTag和标定板
    """
    # 创建一个空白图像
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 绘制模拟的圆点网格（简化版）
    grid_rows, grid_cols = 10, 10
    start_x, start_y = 200, 150
    spacing = 30
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            x = start_x + j * spacing
            y = start_y + i * spacing
            cv2.circle(img, (x, y), 8, (0, 0, 0), -1)
    
    # 绘制模拟的AprilTag位置（简化版）
    tag_x, tag_y = 350, 80
    tag_size = 40
    
    # 绘制AprilTag外框
    cv2.rectangle(img, 
                 (tag_x - tag_size//2, tag_y - tag_size//2),
                 (tag_x + tag_size//2, tag_y + tag_size//2),
                 (0, 0, 0), 2)
    
    # 添加文字说明
    cv2.putText(img, "AprilTag Position", (tag_x - 60, tag_y - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "Calibration Board", (start_x, start_y - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 绘制坐标轴说明
    cv2.arrowedLine(img, (50, 550), (100, 550), (0, 0, 255), 2)
    cv2.arrowedLine(img, (50, 550), (50, 500), (0, 255, 0), 2)
    cv2.putText(img, "X", (105, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "Y", (45, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return img


def demonstrate_coordinate_system_concept():
    """
    演示AprilTag坐标系建立的概念
    """
    print("AprilTag坐标系建立概念演示")
    print("=" * 50)
    
    # 创建演示图像
    demo_img = create_demo_image_with_apriltag()
    
    # 显示概念图
    cv2.imshow('AprilTag坐标系概念图', demo_img)
    print("显示概念图...")
    print("按任意键继续...")
    cv2.waitKey(0)
    
    # 添加说明文字
    explanation_img = demo_img.copy()
    
    explanations = [
        "1. AprilTag提供方向参考",
        "2. 找到最近的标定板角点作为原点",
        "3. X轴方向 = AprilTag正方向",
        "4. Y轴垂直于X轴",
        "5. 重新排列所有角点"
    ]
    
    for i, text in enumerate(explanations):
        cv2.putText(explanation_img, text, (20, 50 + i * 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow('AprilTag坐标系建立步骤', explanation_img)
    print("显示建立步骤...")
    print("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demonstrate_advantages():
    """
    演示AprilTag方法的优势
    """
    print("\nAprilTag方法优势演示")
    print("=" * 50)
    
    # 创建对比图像
    img_width, img_height = 800, 400
    comparison_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # 左侧：传统方法（坐标轴不确定）
    cv2.putText(comparison_img, "Traditional Method", (50, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(comparison_img, "Problems:", (50, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    problems = [
        "- Coordinate axis uncertainty",
        "- 90° rotation ambiguity", 
        "- Inconsistent origin",
        "- Unstable angle measurement"
    ]
    
    for i, problem in enumerate(problems):
        cv2.putText(comparison_img, problem, (50, 90 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # 右侧：AprilTag方法
    cv2.putText(comparison_img, "AprilTag Method", (450, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(comparison_img, "Advantages:", (450, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    advantages = [
        "- Unified coordinate system",
        "- No rotation ambiguity",
        "- Fixed origin reference",
        "- Stable angle measurement"
    ]
    
    for i, advantage in enumerate(advantages):
        cv2.putText(comparison_img, advantage, (450, 90 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # 绘制分隔线
    cv2.line(comparison_img, (400, 0), (400, img_height), (128, 128, 128), 2)
    
    cv2.imshow('方法对比', comparison_img)
    print("显示方法对比...")
    print("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demonstrate_implementation_steps():
    """
    演示实现步骤
    """
    print("\n实现步骤演示")
    print("=" * 50)
    
    steps = [
        "步骤1: 检测AprilTag",
        "步骤2: 检测标定板圆点",
        "步骤3: 找到最近角点作为原点", 
        "步骤4: 建立坐标轴方向",
        "步骤5: 重新排列角点",
        "步骤6: 计算相机倾斜角"
    ]
    
    for i, step in enumerate(steps):
        print(f"{i+1}. {step}")
        
        # 创建步骤说明图
        step_img = np.ones((300, 600, 3), dtype=np.uint8) * 255
        cv2.putText(step_img, step, (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('实现步骤', step_img)
        cv2.waitKey(1000)  # 显示1秒
    
    cv2.destroyAllWindows()


def show_usage_examples():
    """
    显示使用示例
    """
    print("\n使用示例")
    print("=" * 50)
    
    examples = [
        "# 测试单张图像",
        "python test_apriltag_detection.py --image data/board1.png",
        "",
        "# 处理rosbag",
        "python src/tilt_checker_with_apriltag.py \\",
        "    --rosbag rosbags/your_bag \\",
        "    --image-topic /camera/image_raw \\",
        "    --save-images",
        "",
        "# 实时处理",
        "python src/tilt_checker_with_apriltag.py \\",
        "    --image-topic /camera/image_raw \\",
        "    --camera-yaml config/camera_info.yaml"
    ]
    
    for example in examples:
        print(example)


def main():
    """
    主演示函数
    """
    print("AprilTag坐标系建立方案演示")
    print("=" * 60)
    print()
    print("这个演示将展示如何使用AprilTag解决相机倾斜检测中的坐标轴不确定性问题。")
    print()
    
    try:
        # 1. 概念演示
        demonstrate_coordinate_system_concept()
        
        # 2. 优势对比
        demonstrate_advantages()
        
        # 3. 实现步骤
        demonstrate_implementation_steps()
        
        # 4. 使用示例
        show_usage_examples()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print()
        print("接下来您可以：")
        print("1. 准备包含AprilTag的标定板")
        print("2. 使用 test_apriltag_detection.py 测试检测功能")
        print("3. 使用 tilt_checker_with_apriltag.py 处理实际数据")
        print()
        print("详细使用说明请参考：docs/apriltag_coordinate_system_usage.md")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请确保已安装所需依赖：pip install opencv-python numpy")


if __name__ == '__main__':
    main()