#!/usr/bin/env python3
"""
畸变矫正演示工具

功能：
1. 使用真实内参进行畸变矫正
2. 使用默认内参进行畸变矫正（对比）
3. 可视化矫正前后的效果，观察直线是否变直
"""

import cv2
import numpy as np
import argparse
import os
import sys

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_camera_intrinsics, default_intrinsics, get_camera_intrinsics


def draw_grid_lines(img, color=(0, 255, 0), thickness=2, spacing=50):
    """在图像上绘制网格线，用于观察畸变矫正效果"""
    h, w = img.shape[:2]
    vis = img.copy()
    
    # 绘制垂直线
    for x in range(0, w, spacing):
        cv2.line(vis, (x, 0), (x, h), color, thickness)
    
    # 绘制水平线
    for y in range(0, h, spacing):
        cv2.line(vis, (0, y), (w, y), color, thickness)
    
    return vis


def undistort_and_compare(img_path, camera_yaml_path=None, save_dir="outputs"):
    """
    对图像进行畸变矫正并对比效果
    
    参数:
        img_path: 输入图像路径
        camera_yaml_path: 相机内参 YAML 文件路径（可选）
        save_dir: 保存目录
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    
    h, w = img.shape[:2]
    print(f"图像尺寸: {w} x {h}")
    
    # 1. 使用真实内参（如果可用）
    K_real, dist_real, image_size = load_camera_intrinsics(camera_yaml_path or 'config/camera_info.yaml')
    
    if K_real is not None and dist_real is not None:
        print("\n" + "="*60)
        print("使用真实内参进行畸变矫正")
        print("="*60)
        print(f"内参矩阵 K:\n{K_real}")
        print(f"畸变系数 D: {dist_real}")
        
        # 如果图像尺寸不匹配，缩放内参
        if image_size and (image_size[0] != w or image_size[1] != h):
            from src.utils import scale_camera_intrinsics
            print(f"\n[INFO] 图像尺寸不匹配，缩放内参...")
            K_real, dist_real = scale_camera_intrinsics(K_real, dist_real, image_size, (w, h))
            print(f"缩放后的内参矩阵 K:\n{K_real}")
        
        # 畸变矫正
        undistorted_real = cv2.undistort(img, K_real, dist_real)
        
        # 绘制网格线用于观察
        img_with_grid = draw_grid_lines(img)
        undistorted_real_with_grid = draw_grid_lines(undistorted_real)
        
        # 保存结果
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        cv2.imwrite(f"{save_dir}/{base_name}_original_with_grid.png", img_with_grid)
        cv2.imwrite(f"{save_dir}/{base_name}_undistorted_real_with_grid.png", undistorted_real_with_grid)
        cv2.imwrite(f"{save_dir}/{base_name}_undistorted_real.png", undistorted_real)
        
        print(f"\n✅ 已保存:")
        print(f"  - {save_dir}/{base_name}_original_with_grid.png (原始图像+网格)")
        print(f"  - {save_dir}/{base_name}_undistorted_real_with_grid.png (真实内参矫正+网格)")
        print(f"  - {save_dir}/{base_name}_undistorted_real.png (真实内参矫正)")
    else:
        print("\n[WARN] 无法加载真实内参，跳过真实内参矫正")
        K_real = None
        dist_real = None
        undistorted_real = None
    
    # 2. 使用默认内参（对比）
    print("\n" + "="*60)
    print("使用默认内参进行畸变矫正（对比）")
    print("="*60)
    K_default, dist_default = default_intrinsics(h, w, f_scale=1.0)
    print(f"默认内参矩阵 K:\n{K_default}")
    print(f"默认畸变系数 D: {dist_default}")
    print("\n[注意] 默认内参假设无畸变（dist=0），所以矫正后图像不变")
    
    # 使用默认内参进行"矫正"（实际上不会改变，因为 dist=0）
    undistorted_default = cv2.undistort(img, K_default, dist_default)
    
    # 绘制网格线
    undistorted_default_with_grid = draw_grid_lines(undistorted_default)
    
    # 保存结果
    if K_real is not None:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(f"{save_dir}/{base_name}_undistorted_default_with_grid.png", undistorted_default_with_grid)
        print(f"\n✅ 已保存:")
        print(f"  - {save_dir}/{base_name}_undistorted_default_with_grid.png (默认内参矫正+网格)")
    
    # 3. 创建对比图
    if K_real is not None:
        print("\n" + "="*60)
        print("创建对比图")
        print("="*60)
        
        # 水平拼接：原始 | 真实内参矫正
        comparison = np.hstack([img_with_grid, undistorted_real_with_grid])
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(f"{save_dir}/{base_name}_comparison.png", comparison)
        print(f"✅ 已保存对比图: {save_dir}/{base_name}_comparison.png")
        print("   左侧: 原始图像（有畸变）")
        print("   右侧: 真实内参矫正后（无畸变）")
        print("\n💡 观察要点:")
        print("   - 原始图像中的直线（网格线）应该是弯曲的（桶形或枕形畸变）")
        print("   - 矫正后的图像中，直线应该变直")
        print("   - 如果矫正后直线仍然弯曲，说明内参不准确")


def explain_intrinsics():
    """解释内参的含义"""
    print("\n" + "="*60)
    print("相机内参 K 和畸变系数 dist 的含义")
    print("="*60)
    
    print("\n1. 内参矩阵 K (3×3):")
    print("""
    K = [fx  0  cx]
        [ 0 fy  cy]
        [ 0  0   1]
    
    含义:
    - fx, fy: 焦距（像素单位），表示相机在 X 和 Y 方向的放大倍数
    - cx, cy: 主点坐标（像素），表示光轴与图像平面的交点
    - 0, 1: 固定值，用于齐次坐标变换
    
    作用:
    - 将 3D 世界坐标投影到 2D 图像坐标
    - 描述相机的成像特性（视野角度、图像中心等）
    """)
    
    print("\n2. 畸变系数 dist (通常 5 个参数):")
    print("""
    dist = [k1, k2, p1, p2, k3]
    
    含义:
    - k1, k2, k3: 径向畸变系数（radial distortion）
      * 描述镜头中心到边缘的畸变程度
      * k1 > 0: 桶形畸变（边缘向内弯曲）
      * k1 < 0: 枕形畸变（边缘向外弯曲）
    - p1, p2: 切向畸变系数（tangential distortion）
      * 描述镜头与图像平面不平行造成的畸变
    
    作用:
    - 描述镜头的物理缺陷导致的图像畸变
    - 用于矫正图像，使直线变直
    """)
    
    print("\n3. default_intrinsics() 返回的 K, dist:")
    print("""
    - K: 近似内参，假设：
      * fx = fy = max(w, h) * f_scale（焦距近似为图像对角线）
      * cx = w/2, cy = h/2（主点在图像中心）
    - dist: 全零 [0, 0, 0, 0, 0]（假设无畸变）
    
    用途:
    - 当没有真实内参时的近似值
    - 仅用于粗略的位姿估计
    - 不能用于精确的畸变矫正
    """)
    
    print("\n4. 真实 K, dist:")
    print("""
    - 通过相机标定获得（如使用 OpenCV 的 camera_calibration 工具）
    - 或从 ROS2 CameraInfo 话题提取
    - 精确描述相机的成像特性
    
    作用:
    - 精确的 3D-2D 投影
    - 精确的畸变矫正（cv2.undistort）
    - 准确的位姿估计（PnP）
    """)


def explain_projection_matrix():
    """解释投影矩阵 P 的含义和应用"""
    print("\n" + "="*60)
    print("投影矩阵 P (3×4) 的含义和应用")
    print("="*60)
    
    print("""
    投影矩阵 P 的格式:
    P = [fx  0  cx  tx]
        [ 0 fy  cy  ty]
        [ 0  0   1   0]
    
    其中:
    - fx, fy, cx, cy: 与内参矩阵 K 相同
    - tx, ty: 平移参数（通常用于双目视觉或已矫正的图像）
    
    应用场景:
    
    1. 双目视觉（Stereo Vision）:
       - P_left, P_right: 左右相机的投影矩阵
       - 用于将 3D 点投影到左右图像平面
       - 计算视差（disparity）和深度
    
    2. 已矫正图像（Rectified Image）:
       - 经过立体矫正后的图像
       - P 矩阵包含了矫正后的内参
       - 可以直接用于立体匹配
    
    3. 3D 点投影:
       - 将 3D 点 [X, Y, Z, 1] 投影到 2D 图像坐标 [u, v, w]
       - [u, v, w]^T = P × [X, Y, Z, 1]^T
       - 像素坐标: (u/w, v/w)
    
    与内参矩阵 K 的关系:
    - 如果 tx = ty = 0，则 P 的前 3×3 部分等于 K
    - P 可以看作是 K 的扩展，增加了平移项
    
    在本项目中的应用:
    - 通常使用 K 和 dist 进行畸变矫正和 PnP
    - P 矩阵主要用于双目视觉系统
    - 如果使用已矫正的图像，可以用 P 的前 3×3 部分作为 K
    """)


def main():
    parser = argparse.ArgumentParser(description="畸变矫正演示工具")
    parser.add_argument("--image", help="输入图像路径")
    parser.add_argument("--camera-yaml", type=str, default="config/camera_info.yaml",
                       help="相机内参 YAML 文件路径")
    parser.add_argument("--save-dir", type=str, default="outputs",
                       help="保存目录")
    parser.add_argument("--explain", action="store_true",
                       help="显示内参和投影矩阵的详细说明")
    args = parser.parse_args()
    
    if args.explain:
        explain_intrinsics()
        explain_projection_matrix()
        print("\n" + "="*60)
    
    if args.image:
        undistort_and_compare(args.image, args.camera_yaml, args.save_dir)
    elif not args.explain:
        parser.print_help()
        print("\n提示: 使用 --explain 查看详细说明，或使用 --image 处理图像")


if __name__ == "__main__":
    main()

