#!/usr/bin/env python3
"""
相机标定和重投影误差分析工具

功能：
1. 使用 YAML 中的内参对 data 目录中的所有图像进行畸变矫正
2. 使用矫正后的图像进行相机标定
3. 计算所有图像的重投影误差
4. 绘制重投影误差分布散点图
"""

import cv2
import numpy as np
import os
import sys
import glob
import yaml
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

# 设置 matplotlib 支持中文（如果系统有中文字体）
try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass  # 如果设置失败，使用默认字体

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_camera_intrinsics, scale_camera_intrinsics
from src.detect_grid_improved import try_find_adaptive, refine
from src.estimate_tilt import build_obj_points


def undistort_images(data_dir, output_dir, camera_yaml_path='config/camera_info.yaml'):
    """
    对 data 目录中的所有图像进行畸变矫正
    
    参数:
        data_dir: 输入图像目录
        output_dir: 输出目录（矫正后的图像）
        camera_yaml_path: 相机内参 YAML 文件路径
    
    返回:
        undistorted_images: 矫正后的图像列表（用于后续标定）
        image_paths: 原始图像路径列表
    """
    print("="*60)
    print("步骤 1: 畸变矫正")
    print("="*60)
    
    # 加载内参
    K, dist, image_size = load_camera_intrinsics(camera_yaml_path)
    if K is None or dist is None:
        raise RuntimeError(f"无法加载相机内参: {camera_yaml_path}")
    
    print(f"加载的内参矩阵 K:\n{K}")
    print(f"畸变系数 D: {dist}")
    if image_size:
        print(f"YAML 中记录的图像尺寸: {image_size[0]} x {image_size[1]}")
    
    # 查找所有图像文件
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
    image_paths = sorted(image_paths)
    
    if not image_paths:
        raise FileNotFoundError(f"在 {data_dir} 中未找到图像文件")
    
    print(f"\n找到 {len(image_paths)} 张图像")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储矫正后的图像和路径
    undistorted_images = []
    undistorted_paths = []
    
    # 处理每张图像
    for i, img_path in enumerate(image_paths):
        print(f"\n处理 [{i+1}/{len(image_paths)}]: {os.path.basename(img_path)}")
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] 无法读取图像，跳过")
            continue
        
        h, w = img.shape[:2]
        print(f"  图像尺寸: {w} x {h}")
        
        # 如果图像尺寸不匹配，缩放内参
        K_used = K.copy()
        dist_used = dist.copy()
        if image_size and (image_size[0] != w or image_size[1] != h):
            print(f"  [INFO] 图像尺寸不匹配，缩放内参...")
            K_used, dist_used = scale_camera_intrinsics(K, dist, image_size, (w, h))
        
        # 畸变矫正
        undistorted = cv2.undistort(img, K_used, dist_used)
        
        # 保存矫正后的图像
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        output_path = os.path.join(output_dir, f"{base_name}_undistorted{ext}")
        cv2.imwrite(output_path, undistorted)
        
        undistorted_images.append(undistorted)
        undistorted_paths.append(output_path)
        print(f"  ✅ 已保存: {output_path}")
    
    print(f"\n✅ 完成！共矫正 {len(undistorted_images)} 张图像")
    return undistorted_images, undistorted_paths, image_paths


def calibrate_camera(undistorted_images, undistorted_paths, rows=15, cols=15, spacing=10.0):
    """
    使用矫正后的图像进行相机标定
    
    参数:
        undistorted_images: 矫正后的图像列表
        undistorted_paths: 矫正后的图像路径列表
        rows: 圆点行数
        cols: 圆点列数
        spacing: 圆点间距（mm）
    
    返回:
        camera_matrix: 标定得到的内参矩阵
        dist_coeffs: 标定得到的畸变系数
        rvecs: 旋转向量列表
        tvecs: 平移向量列表
        objpoints: 3D 点列表
        imgpoints: 2D 点列表
        calibration_images: 成功标定的图像索引列表
    """
    print("\n" + "="*60)
    print("步骤 2: 相机标定")
    print("="*60)
    
    # 准备 3D 对象点（世界坐标系）
    objp = build_obj_points(rows, cols, spacing, symmetric=True)
    print(f"3D 对象点: {objp.shape[0]} 个点，间距 {spacing}mm")
    
    # 存储所有图像的 3D 和 2D 点
    objpoints = []  # 3D 点（世界坐标系）
    imgpoints = []  # 2D 点（图像坐标系）
    calibration_images = []  # 成功标定的图像索引
    
    # 检测每张图像的圆点
    print(f"\n检测圆点网格...")
    for i, (img, img_path) in enumerate(zip(undistorted_images, undistorted_paths)):
        print(f"\n处理 [{i+1}/{len(undistorted_images)}]: {os.path.basename(img_path)}")
        
        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 检测圆点网格（使用 try_find_adaptive 直接检测指定尺寸）
        ok, corners, _ = try_find_adaptive(gray, rows, cols, symmetric=True)
        
        if not ok or corners is None:
            print(f"  [WARN] 未检测到 {rows}×{cols} 网格，跳过")
            continue
        
        if len(corners) != rows * cols:
            print(f"  [WARN] 检测到 {len(corners)} 个点，期望 {rows*cols} 个，跳过")
            continue
        
        # 精化角点位置
        corners_refined = refine(gray, corners)
        
        # 添加到列表
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        calibration_images.append(i)
        
        print(f"  ✅ 成功检测到 {len(corners_refined)} 个点")
    
    if len(objpoints) < 3:
        raise RuntimeError(f"成功检测的图像数量不足（{len(objpoints)} 张），至少需要 3 张图像进行标定")
    
    print(f"\n✅ 成功检测 {len(objpoints)} 张图像，开始标定...")
    
    # 获取图像尺寸
    h, w = undistorted_images[0].shape[:2]
    
    # 进行相机标定
    print(f"\n执行相机标定（图像尺寸: {w} x {h}）...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None,
        flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3
    )
    
    if not ret:
        raise RuntimeError("相机标定失败")
    
    print(f"\n✅ 标定完成！")
    print(f"标定得到的内参矩阵 K:\n{camera_matrix}")
    print(f"标定得到的畸变系数 D: {dist_coeffs.flatten()}")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs, objpoints, imgpoints, calibration_images


def calculate_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    计算重投影误差
    
    参数:
        objpoints: 3D 点列表
        imgpoints: 2D 点列表
        rvecs: 旋转向量列表
        tvecs: 平移向量列表
        camera_matrix: 内参矩阵
        dist_coeffs: 畸变系数
    
    返回:
        errors_per_image: 每张图像的平均误差列表
        all_errors: 所有点的误差列表（标量：像素）
        residual_vectors: 所有点的残差向量列表（dx, dy），用于零中心散点图
    """
    print("\n" + "="*60)
    print("步骤 3: 计算重投影误差")
    print("="*60)
    
    total_error = 0
    total_points = 0
    errors_per_image = []
    all_errors = []
    residual_vectors = []
    
    for i in range(len(objpoints)):
        # 将 3D 点投影到 2D 图像平面
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], 
            camera_matrix, dist_coeffs
        )
        
        # 计算误差
        imgpoints2 = imgpoints2.reshape(-1, 2)
        imgpoints_i = imgpoints[i].reshape(-1, 2)
        
        # 计算每个点的误差
        residuals = imgpoints_i - imgpoints2  # 形状 (N, 2) -> (dx, dy)
        errors = np.linalg.norm(residuals, axis=1)
        all_errors.extend(errors.tolist())
        residual_vectors.extend(residuals.tolist())
        
        # 计算平均误差
        error = np.mean(errors)
        errors_per_image.append(error)
        
        total_error += error * len(imgpoints2)
        total_points += len(imgpoints2)
        
        print(f"图像 {i+1}: 平均误差 = {error:.4f} 像素, 点数 = {len(imgpoints2)}")
    
    mean_error = total_error / total_points
    print(f"\n✅ 总平均重投影误差: {mean_error:.4f} 像素")
    print(f"   误差范围: {min(errors_per_image):.4f} ~ {max(errors_per_image):.4f} 像素")
    
    return errors_per_image, all_errors, residual_vectors


def plot_reprojection_errors(errors_per_image, all_errors, save_path='outputs/reprojection_errors.png'):
    """
    绘制重投影误差分布散点图
    
    参数:
        errors_per_image: 每张图像的平均误差列表
        all_errors: 所有点的误差列表
        save_path: 保存路径
    """
    print("\n" + "="*60)
    print("步骤 4: 绘制重投影误差分布图")
    print("="*60)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 每张图像的平均误差散点图
    ax1 = axes[0]
    image_indices = range(1, len(errors_per_image) + 1)
    ax1.scatter(image_indices, errors_per_image, alpha=0.6, s=100, c='blue', edgecolors='black')
    mean_err = np.mean(errors_per_image)
    ax1.axhline(y=mean_err, color='r', linestyle='--', 
                label=f'Mean Error: {mean_err:.4f} pixels')
    ax1.set_xlabel('Image Index', fontsize=12)
    ax1.set_ylabel('Mean Reprojection Error (pixels)', fontsize=12)
    ax1.set_title('Mean Reprojection Error per Image', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加数值标注
    for i, err in enumerate(errors_per_image):
        ax1.annotate(f'{err:.3f}', (i+1, err), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    # 图2: 所有点的误差分布直方图
    ax2 = axes[1]
    ax2.hist(all_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    mean_all = np.mean(all_errors)
    median_all = np.median(all_errors)
    ax2.axvline(x=mean_all, color='r', linestyle='--', 
                label=f'Mean: {mean_all:.4f} pixels')
    ax2.axvline(x=median_all, color='orange', linestyle='--', 
                label=f'Median: {median_all:.4f} pixels')
    ax2.set_xlabel('Reprojection Error (pixels)', fontsize=12)
    ax2.set_ylabel('Number of Points', fontsize=12)
    ax2.set_title('Reprojection Error Distribution (All Points)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存误差分布图: {save_path}")
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"  图像数量: {len(errors_per_image)}")
    print(f"  总点数: {len(all_errors)}")
    print(f"  每张图像平均误差: {np.mean(errors_per_image):.4f} ± {np.std(errors_per_image):.4f} 像素")
    print(f"  所有点平均误差: {np.mean(all_errors):.4f} ± {np.std(all_errors):.4f} 像素")
    print(f"  最小误差: {np.min(all_errors):.4f} 像素")
    print(f"  最大误差: {np.max(all_errors):.4f} 像素")
    print(f"  中位数误差: {np.median(all_errors):.4f} 像素")
    
    plt.close()


def plot_residual_scatter(residual_vectors, save_path='outputs/reprojection_residual_scatter.png'):
    """
    绘制零中心的重投影误差残差散点图（dx, dy）
    
    参数:
        residual_vectors: [(dx, dy), ...] 列表，单位像素
        save_path: 保存路径
    """
    print("\n" + "="*60)
    print("步骤 5: 绘制零中心残差散点图")
    print("="*60)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    residuals = np.asarray(residual_vectors, dtype=np.float32)
    if residuals.ndim != 2 or residuals.shape[1] != 2:
        raise ValueError("residual_vectors 形状应为 (N, 2)")
    
    dx = residuals[:, 0]
    dy = residuals[:, 1]
    mags = np.hypot(dx, dy)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(dx, dy, c=mags, cmap='viridis', s=16, alpha=0.8, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Residual Magnitude (pixels)')
    
    ax.axhline(0, color='gray', linewidth=1, alpha=0.6)
    ax.axvline(0, color='gray', linewidth=1, alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('dx (pixels)')
    ax.set_ylabel('dy (pixels)')
    ax.set_title('Reprojection Residuals (Zero-centered)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存残差散点图: {save_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="相机标定和重投影误差分析")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="输入图像目录")
    parser.add_argument("--undistorted-dir", type=str, default="data_undistorted",
                       help="矫正后的图像保存目录")
    parser.add_argument("--camera-yaml", type=str, default="config/camera_info.yaml",
                       help="相机内参 YAML 文件路径")
    parser.add_argument("--rows", type=int, default=15,
                       help="圆点行数（默认15）")
    parser.add_argument("--cols", type=int, default=15,
                       help="圆点列数（默认15）")
    parser.add_argument("--spacing", type=float, default=10.0,
                       help="圆点间距（mm，默认10.0）")
    parser.add_argument("--output", type=str, default="outputs/reprojection_errors.png",
                       help="误差分布图保存路径")
    
    args = parser.parse_args()
    
    try:
        # 步骤 1: 畸变矫正
        undistorted_images, undistorted_paths, original_paths = undistort_images(
            args.data_dir, args.undistorted_dir, args.camera_yaml
        )
        
        # 步骤 2: 相机标定
        camera_matrix, dist_coeffs, rvecs, tvecs, objpoints, imgpoints, calibration_images = calibrate_camera(
            undistorted_images, undistorted_paths, args.rows, args.cols, args.spacing
        )
        
        # 步骤 3: 计算重投影误差
        errors_per_image, all_errors, residual_vectors = calculate_reprojection_errors(
            objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
        )
        
        # 步骤 4: 绘制误差分布图
        plot_reprojection_errors(errors_per_image, all_errors, args.output)
        
        # 额外：绘制零中心残差散点图，并导出 CSV
        scatter_path = os.path.join(os.path.dirname(args.output), "reprojection_residual_scatter.png")
        plot_residual_scatter(residual_vectors, scatter_path)
        csv_path = os.path.join(os.path.dirname(args.output), "reprojection_residuals.csv")
        try:
            import csv
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["dx_pixels", "dy_pixels", "magnitude_pixels"])
                for dx, dy in residual_vectors:
                    mag = float(np.hypot(dx, dy))
                    writer.writerow([float(dx), float(dy), mag])
            print(f"✅ 已导出残差 CSV: {csv_path}")
        except Exception as e:
            print(f"[WARN] 导出残差 CSV 失败: {e}")
        
        print("\n" + "="*60)
        print("✅ 全部完成！")
        print("="*60)
        print(f"矫正后的图像保存在: {args.undistorted_dir}")
        print(f"误差分布图保存在: {args.output}")
        print(f"残差散点图保存在: {scatter_path}")
        print(f"残差 CSV 保存在: {csv_path}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

