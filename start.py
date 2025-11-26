#!/usr/bin/env python3
"""
一键启动脚本 - 鲁棒AprilTag系统

解决247像素重投影误差问题的完整解决方案
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查依赖项"""
    print("🔍 检查依赖项...")
    
    required_packages = [
        'cv2', 'numpy', 'pupil_apriltags'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'pupil_apriltags':
                import pupil_apriltags
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install opencv-python numpy pupil-apriltags")
        return False
    
    print("✅ 所有依赖项已安装")
    return True


def check_data_files():
    """检查数据文件"""
    print("📁 检查数据文件...")
    
    # 检查相机参数文件
    camera_yaml = Path('config/camera_info.yaml')
    if not camera_yaml.exists():
        print(f"❌ 相机参数文件不存在: {camera_yaml}")
        print("请确保相机已标定并生成参数文件")
        return False
    
    # 检查数据目录
    data_dir = Path('data')
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    # 检查图像文件
    image_files = list(data_dir.glob('*.png')) + list(data_dir.glob('*.jpg'))
    if not image_files:
        print(f"❌ 数据目录中没有图像文件: {data_dir}")
        return False
    
    print(f"✅ 找到 {len(image_files)} 张图像文件")
    return True


def show_menu():
    """显示菜单"""
    print("\n" + "="*60)
    print("🚀 鲁棒AprilTag系统 - 解决PnP多解歧义问题")
    print("="*60)
    print("选择运行模式:")
    print("1. 🔍 全面AprilTag家族测试 (找到正确的AprilTag类型)")
    print("2. 🎯 运行鲁棒AprilTag系统 (解决247px误差问题)")
    print("3. 📊 演示PnP解决方案效果")
    print("4. 📖 查看快速修复指南")
    print("5. ❌ 退出")
    print("="*60)


def run_apriltag_family_test():
    """运行AprilTag家族测试"""
    print("\n🔍 启动AprilTag家族测试...")
    
    # 选择测试图像
    data_dir = Path('data')
    image_files = list(data_dir.glob('*.png')) + list(data_dir.glob('*.jpg'))
    
    if not image_files:
        print("❌ 没有找到测试图像")
        return
    
    # 使用第一张图像进行测试
    test_image = str(image_files[0])
    print(f"使用测试图像: {test_image}")
    
    cmd = [
        sys.executable, 'comprehensive_apriltag_test.py',
        '--image', test_image,
        '--camera-yaml', 'config/camera_info.yaml'
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ AprilTag家族测试失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到comprehensive_apriltag_test.py文件")


def run_robust_system():
    """运行鲁棒AprilTag系统"""
    print("\n🎯 启动鲁棒AprilTag系统...")
    
    # 基本参数
    cmd = [
        sys.executable, 'run_robust_apriltag_system.py',
        '--data-dir', 'data',
        '--camera-yaml', 'config/camera_info.yaml',
        '--tag-family', 'tagStandard41h12',
        '--max-error', '10.0'
    ]
    
    print("使用参数:")
    print(f"  数据目录: data")
    print(f"  相机参数: config/camera_info.yaml")
    print(f"  AprilTag家族: tagStandard41h12")
    print(f"  最大误差阈值: 10.0px")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 鲁棒系统运行失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到run_robust_apriltag_system.py文件")


def show_demo():
    """显示演示"""
    print("\n📊 启动PnP解决方案演示...")
    
    cmd = [sys.executable, 'demo_pnp_solution.py']
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 演示运行失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到demo_pnp_solution.py文件")


def show_quick_guide():
    """显示快速指南"""
    guide_file = Path('QUICK_FIX_GUIDE.md')
    
    if guide_file.exists():
        print("\n📖 快速修复指南:")
        print("="*60)
        with open(guide_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 只显示前面的关键部分
            lines = content.split('\n')
            for i, line in enumerate(lines[:50]):  # 显示前50行
                print(line)
            
            if len(lines) > 50:
                print("\n... (更多内容请查看 QUICK_FIX_GUIDE.md 文件)")
    else:
        print("❌ 找不到快速指南文件")


def main():
    """主函数"""
    print("🔧 鲁棒AprilTag系统启动器")
    
    # 检查依赖项
    if not check_dependencies():
        return
    
    # 检查数据文件
    if not check_data_files():
        print("\n💡 建议:")
        print("1. 确保相机已标定: python src/calibration_and_reprojection.py")
        print("2. 确保data目录中有测试图像")
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("\n请选择 (1-5): ").strip()
            
            if choice == '1':
                run_apriltag_family_test()
            elif choice == '2':
                run_robust_system()
            elif choice == '3':
                show_demo()
            elif choice == '4':
                show_quick_guide()
            elif choice == '5':
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择，请输入1-5")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
        
        input("\n按回车键继续...")


if __name__ == '__main__':
    main()