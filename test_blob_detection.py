#!/usr/bin/env python3
"""
测试Blob检测参数的脚本
用于快速调试为什么某些帧检测不到圆点
"""

import cv2
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import build_blob_detector, BLOB_DETECTOR_PARAMS


def test_blob_detection_on_image(image_path):
    """测试单张图像的Blob检测"""
    
    print(f"\n{'='*80}")
    print(f"测试图像: {image_path}")
    print(f"{'='*80}\n")
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    # 转灰度
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    print(f"图像尺寸: {img.shape[1]}x{img.shape[0]}")
    print(f"灰度范围: [{gray.min()}, {gray.max()}]")
    print(f"平均亮度: {gray.mean():.1f}")
    print(f"\n当前Blob检测参数:")
    for key, value in BLOB_DETECTOR_PARAMS.items():
        print(f"  {key}: {value}")
    
    # 创建Blob检测器
    detector = build_blob_detector()
    
    # 检测
    keypoints = detector.detect(gray)
    
    print(f"\n🔍 检测结果: 找到 {len(keypoints)} 个候选圆点")
    
    if len(keypoints) == 0:
        print("\n❌ 未检测到任何圆点！")
        print("\n可能的原因:")
        print("  1. 图像太暗或太亮")
        print("  2. 圆点面积超出范围 [minArea, maxArea]")
        print("  3. 圆点形状不符合圆度/惯性比要求")
        print("  4. 阈值范围不合适")
        
        print("\n建议:")
        print("  1. 检查图像质量（亮度、对比度）")
        print("  2. 调整 minArea 和 maxArea")
        print("  3. 降低 minCircularity 和 minInertiaRatio")
        print("  4. 调整 minThreshold 和 maxThreshold")
    else:
        print(f"\n✅ 检测成功！")
        
        # 统计圆点大小
        sizes = [kp.size for kp in keypoints]
        areas = [np.pi * (s/2)**2 for s in sizes]
        
        print(f"\n圆点统计:")
        print(f"  大小范围: [{min(sizes):.1f}, {max(sizes):.1f}]")
        print(f"  面积范围: [{min(areas):.1f}, {max(areas):.1f}]")
        print(f"  平均大小: {np.mean(sizes):.1f}")
        print(f"  平均面积: {np.mean(areas):.1f}")
        
        # 检查是否有圆点超出参数范围
        out_of_range = [a for a in areas if a < BLOB_DETECTOR_PARAMS["minArea"] or a > BLOB_DETECTOR_PARAMS["maxArea"]]
        if out_of_range:
            print(f"\n⚠️ 警告: 有 {len(out_of_range)} 个圆点的面积超出参数范围")
    
    # 可视化
    vis = img.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)
        cv2.circle(vis, (x, y), size//2, (0, 255, 0), 2)
        cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
    
    # 添加信息
    cv2.putText(vis, f'Blobs: {len(keypoints)}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 保存结果
    output_path = image_path.replace('.png', '_blob_test.png')
    cv2.imwrite(output_path, vis)
    print(f"\n💾 已保存可视化结果: {output_path}")
    
    return len(keypoints)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python test_blob_detection.py <图像路径>")
        print("示例: python test_blob_detection.py outputs/robust_apriltag_recording_final_result/images/frame_000006_debug_gray.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在: {image_path}")
        sys.exit(1)
    
    test_blob_detection_on_image(image_path)


if __name__ == '__main__':
    main()
