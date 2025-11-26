#!/usr/bin/env python3
"""
快速测试网格检测功能
用于验证改进后的检测算法
"""
import cv2
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detect_grid_improved import smart_auto_search, try_find_adaptive
from src.utils import build_blob_detector

def test_single_image(image_path):
    """测试单张图像的检测"""
    print(f"\n{'='*60}")
    print(f"测试图像: {image_path}")
    print(f"{'='*60}\n")
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"图像尺寸: {w} x {h}")
    
    # 1. 先尝试直接检测 15x15
    print(f"\n1. 尝试直接检测 15x15 对称网格...")
    ok, centers, keypoints = try_find_adaptive(gray, 15, 15, symmetric=True)
    
    if ok:
        print(f"✅ 成功检测到 15x15 网格 ({len(centers)} 个点)")
        return True
    else:
        print(f"❌ 15x15 检测失败")
    
    # 2. 尝试自动搜索
    print(f"\n2. 尝试自动搜索...")
    ok, corners, meta, keypoints = smart_auto_search(
        gray, 
        rows_range=(10, 16), 
        cols_range=(10, 16),
        max_attempts=30,  # 限制尝试次数
        timeout_seconds=5.0  # 5秒超时
    )
    
    if ok and meta:
        rows, cols, symmetric = meta
        print(f"✅ 自动搜索成功: {rows}x{cols}, symmetric={symmetric}")
        return True
    else:
        print(f"❌ 自动搜索失败")
        
        # 3. 显示 blob 检测结果
        det = build_blob_detector()
        keypoints = det.detect(gray)
        print(f"\n3. Blob 检测结果: {len(keypoints)} 个候选点")
        
        if len(keypoints) > 0:
            print(f"   建议:")
            print(f"   - 检测到的点数量: {len(keypoints)}")
            print(f"   - 估算网格尺寸: {int(len(keypoints)**0.5)}x{int(len(keypoints)**0.5)}")
            print(f"   - 可能原因: 网格结构不完整或透视畸变过大")
        else:
            print(f"   建议:")
            print(f"   - 未检测到任何圆点")
            print(f"   - 可能原因: 光照不足、对比度低、或 blob 参数不合适")
        
        return False

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python test_detection_quick.py <image_path>")
        print("示例: python test_detection_quick.py outputs/testbag_analysis/images/frame_000001_result.png")
        return
    
    image_path = sys.argv[1]
    success = test_single_image(image_path)
    
    if success:
        print(f"\n{'='*60}")
        print(f"✅ 测试通过")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"❌ 测试失败")
        print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
