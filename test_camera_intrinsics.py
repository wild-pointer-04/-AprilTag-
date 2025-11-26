#!/usr/bin/env python3
"""
测试相机内参加载功能

使用方法:
    python test_camera_intrinsics.py
"""

import os
import sys
import numpy as np
import yaml

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_camera_intrinsics, get_camera_intrinsics, default_intrinsics


def test_yaml_loading():
    """测试 YAML 文件加载"""
    print("=" * 60)
    print("测试 1: 加载相机内参 YAML 文件")
    print("=" * 60)
    
    yaml_path = 'config/camera_info.yaml'
    
    # 检查文件是否存在
    if os.path.exists(yaml_path):
        print(f"✅ 找到 YAML 文件: {yaml_path}")
        K, dist, image_size = load_camera_intrinsics(yaml_path)
        
        if K is not None and dist is not None:
            print(f"✅ 成功加载相机内参")
            print(f"  内参矩阵 K:\n{K}")
            print(f"  畸变系数 D: {dist}")
            if image_size:
                print(f"  图像尺寸: {image_size[0]} x {image_size[1]}")
        else:
            print(f"❌ 加载失败")
    else:
        print(f"⚠️  YAML 文件不存在: {yaml_path}")
        print(f"   这是正常的，如果还没有从 ROS2 提取内参")


def test_get_intrinsics():
    """测试 get_camera_intrinsics 函数"""
    print("\n" + "=" * 60)
    print("测试 2: get_camera_intrinsics 函数（自动回退）")
    print("=" * 60)
    
    h, w = 480, 640
    K, dist = get_camera_intrinsics(h, w, yaml_path='config/camera_info.yaml')
    
    print(f"✅ 获取相机内参成功")
    print(f"  内参矩阵 K:\n{K}")
    print(f"  畸变系数 D: {dist}")
    print(f"  图像尺寸: {w} x {h}")


def test_default_intrinsics():
    """测试默认内参"""
    print("\n" + "=" * 60)
    print("测试 3: 默认内参生成")
    print("=" * 60)
    
    h, w = 480, 640
    K, dist = default_intrinsics(h, w, f_scale=1.0)
    
    print(f"✅ 生成默认内参成功")
    print(f"  内参矩阵 K:\n{K}")
    print(f"  畸变系数 D: {dist}")
    print(f"  焦距: {K[0, 0]:.1f} (假设)")
    print(f"  主点: ({K[0, 2]:.1f}, {K[1, 2]:.1f})")


def test_yaml_format():
    """测试 YAML 文件格式（如果存在）"""
    print("\n" + "=" * 60)
    print("测试 4: YAML 文件格式验证")
    print("=" * 60)
    
    yaml_path = 'config/camera_info.yaml'
    
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            print(f"✅ YAML 文件格式正确")
            print(f"  包含的字段: {list(data.keys())}")
            
            # 检查必需字段
            required_fields = ['camera_matrix', 'distortion_coefficients']
            for field in required_fields:
                if field in data:
                    print(f"  ✅ 包含字段: {field}")
                else:
                    print(f"  ❌ 缺少字段: {field}")
            
            # 检查内参矩阵
            if 'camera_matrix' in data:
                cam_matrix = data['camera_matrix']
                if 'data' in cam_matrix:
                    K_data = cam_matrix['data']
                    if len(K_data) == 9:
                        print(f"  ✅ 内参矩阵数据长度正确: {len(K_data)}")
                        K = np.array(K_data).reshape(3, 3)
                        print(f"  内参矩阵:\n{K}")
                    else:
                        print(f"  ❌ 内参矩阵数据长度错误: {len(K_data)} (期望 9)")
            
            # 检查畸变系数
            if 'distortion_coefficients' in data:
                dist_coeffs = data['distortion_coefficients']
                if 'data' in dist_coeffs:
                    dist_data = dist_coeffs['data']
                    print(f"  ✅ 畸变系数数量: {len(dist_data)}")
                    print(f"  畸变系数: {dist_data}")
        except Exception as e:
            print(f"❌ YAML 文件解析失败: {e}")
    else:
        print(f"⚠️  YAML 文件不存在，跳过格式验证")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("相机内参加载功能测试")
    print("=" * 60)
    
    # 运行测试
    test_yaml_loading()
    test_get_intrinsics()
    test_default_intrinsics()
    test_yaml_format()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    main()


