#!/usr/bin/env python3
"""
从2D重投影误差反推3D空间的实际位置误差
用于机械臂标定的误差分析（正确版本）
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import load_camera_intrinsics


def calculate_3d_error_from_reprojection(
    reprojection_error_px,
    camera_matrix,
    distance_z_mm,
    board_rows=15,
    board_cols=15,
    board_spacing_mm=65.0,
    gripper_offset_mm=0.0
):
    """
    从2D重投影误差准确计算3D空间的实际位置误差
    
    基于误差传播理论：
    - 平面误差（X,Y）：直接由像素误差和深度决定
    - 深度误差（Z）：由重投影误差和几何配置共同决定
    - 角度误差：由位置误差和标定板几何约束决定
    
    参数:
        reprojection_error_px: 重投影误差RMS（像素）
        camera_matrix: 相机内参矩阵 K
        distance_z_mm: 相机到标定板的距离（mm）
        board_rows: 标定板行数
        board_cols: 标定板列数
        board_spacing_mm: 标定板点间距（mm）
        gripper_offset_mm: 机械臂末端相对于标定板的偏移（mm）
    
    返回:
        dict: 包含各种误差的字典
    """
    
    # 提取相机参数
    fx = camera_matrix[0, 0]  # X方向焦距（像素）
    fy = camera_matrix[1, 1]  # Y方向焦距（像素）
    cx = camera_matrix[0, 2]  # 主点X坐标
    cy = camera_matrix[1, 2]  # 主点Y坐标
    
    print(f"\n{'='*80}")
    print(f"从2D重投影误差反推3D空间误差（基于误差传播理论）")
    print(f"{'='*80}")
    
    print(f"\n输入参数:")
    print(f"  重投影误差RMS: {reprojection_error_px:.3f} 像素")
    print(f"  工作距离 Z: {distance_z_mm:.1f} mm")
    print(f"  相机焦距 fx: {fx:.2f} 像素, fy: {fy:.2f} 像素")
    print(f"  标定板尺寸: {board_rows}×{board_cols}")
    print(f"  点间距: {board_spacing_mm} mm")
    if gripper_offset_mm != 0:
        print(f"  机械臂偏移: {gripper_offset_mm:.1f} mm")
    
    # ========================================
    # 步骤1: 计算平面位置误差（X-Y平面）
    # ========================================
    # 基于针孔相机模型: Δx = (Z/fx) * Δu, Δy = (Z/fy) * Δv
    # 其中 Δu, Δv 是图像平面的像素误差
    
    error_x_mm = (distance_z_mm / fx) * reprojection_error_px
    error_y_mm = (distance_z_mm / fy) * reprojection_error_px
    
    # 平面误差的2D范数（假设X和Y方向独立）
    error_xy_mm = np.sqrt(error_x_mm**2 + error_y_mm**2)
    
    print(f"\n步骤1: 平面位置误差（标定板平面）")
    print(f"  X方向误差: σ_x = ±{error_x_mm:.3f} mm")
    print(f"  Y方向误差: σ_y = ±{error_y_mm:.3f} mm")
    print(f"  平面总误差: σ_xy = ±{error_xy_mm:.3f} mm")
    
    # ========================================
    # 步骤2: 估算深度误差（Z方向）
    # ========================================
    # 深度误差的估算基于以下考虑：
    # 1. 单目PnP的深度不确定性与以下因素相关：
    #    - 平面位置误差
    #    - 观测点的空间分布（标定板尺寸）
    #    - 观测距离
    #
    # 理论推导（简化）：
    # σ_Z ≈ (Z²/B) * (σ_pixel/f)
    # 其中 B 是有效基线，对于标定板约等于其有效尺寸
    
    # 计算标定板的有效尺寸
    board_width_mm = (board_cols - 1) * board_spacing_mm
    board_height_mm = (board_rows - 1) * board_spacing_mm
    board_diagonal_mm = np.sqrt(board_width_mm**2 + board_height_mm**2)
    
    # 有效基线（取标定板对角线的一半作为保守估计）
    effective_baseline_mm = board_diagonal_mm / 2.0
    
    # 焦距的平均值（用于深度计算）
    f_avg = (fx + fy) / 2.0
    
    # 深度误差估算（基于三角测量误差传播）
    # σ_Z ≈ (Z² / (B * f)) * σ_pixel
    error_z_mm = (distance_z_mm**2 / (effective_baseline_mm * f_avg)) * reprojection_error_px
    
    # 另一种估算方法（更保守）：基于视差角度的不确定性
    # θ = arctan(B/Z), Δθ ≈ σ_pixel/f
    # σ_Z ≈ Z * tan(Δθ) ≈ Z * (σ_pixel/f)
    error_z_conservative_mm = distance_z_mm * (reprojection_error_px / f_avg)
    
    print(f"\n步骤2: 深度误差估算（Z方向）")
    print(f"  标定板尺寸: {board_width_mm:.1f} × {board_height_mm:.1f} mm")
    print(f"  标定板对角线: {board_diagonal_mm:.1f} mm")
    print(f"  有效基线: {effective_baseline_mm:.1f} mm")
    print(f"  深度误差（三角测量）: σ_z = ±{error_z_mm:.3f} mm")
    print(f"  深度误差（保守估计）: σ_z = ±{error_z_conservative_mm:.3f} mm")
    print(f"  采用值: ±{error_z_mm:.3f} mm")
    
    # ========================================
    # 步骤3: 计算3D位置总误差
    # ========================================
    # 假设X, Y, Z方向的误差独立，总误差为：
    # σ_3D = sqrt(σ_x² + σ_y² + σ_z²)
    
    error_3d_mm = np.sqrt(error_x_mm**2 + error_y_mm**2 + error_z_mm**2)
    
    print(f"\n步骤3: 3D位置总误差")
    print(f"  3D总误差: σ_3D = ±{error_3d_mm:.3f} mm")
    print(f"  误差分量: [σ_x={error_x_mm:.3f}, σ_y={error_y_mm:.3f}, σ_z={error_z_mm:.3f}]")
    
    # ========================================
    # 步骤4: 计算旋转角度误差
    # ========================================
    # 姿态误差的估算基于以下考虑：
    # 1. 平移误差会导致姿态估计的偏差
    # 2. 角度误差与位置误差和标定板尺寸有关
    
    # 方法1: 基于位置误差相对于标定板尺寸的比例
    # Δθ ≈ arctan(Δx / L)，其中L是标定板特征尺寸
    angular_error_x_rad = np.arctan(error_x_mm / board_width_mm)
    angular_error_y_rad = np.arctan(error_y_mm / board_height_mm)
    angular_error_x_deg = np.degrees(angular_error_x_rad)
    angular_error_y_deg = np.degrees(angular_error_y_rad)
    
    # 方法2: 基于重投影误差和距离
    # 这给出了绕Z轴旋转的误差
    angular_error_z_rad = reprojection_error_px / f_avg
    angular_error_z_deg = np.degrees(angular_error_z_rad)
    
    # 综合角度误差（取最大值作为保守估计）
    angular_error_max_deg = max(angular_error_x_deg, angular_error_y_deg, angular_error_z_deg)
    
    print(f"\n步骤4: 旋转角度误差")
    print(f"  Roll误差（绕X轴）: σ_θx = ±{angular_error_x_deg:.4f}°")
    print(f"  Pitch误差（绕Y轴）: σ_θy = ±{angular_error_y_deg:.4f}°")
    print(f"  Yaw误差（绕Z轴）: σ_θz = ±{angular_error_z_deg:.4f}°")
    print(f"  最大角度误差: ±{angular_error_max_deg:.4f}°")
    
    # ========================================
    # 步骤5: 计算机械臂末端位置误差
    # ========================================
    if gripper_offset_mm != 0:
        gripper_distance_mm = distance_z_mm + gripper_offset_mm
        
        # 机械臂末端的平面误差
        gripper_error_x_mm = (gripper_distance_mm / fx) * reprojection_error_px
        gripper_error_y_mm = (gripper_distance_mm / fy) * reprojection_error_px
        gripper_error_xy_mm = np.sqrt(gripper_error_x_mm**2 + gripper_error_y_mm**2)
        
        # 机械臂末端的深度误差
        gripper_error_z_mm = (gripper_distance_mm**2 / (effective_baseline_mm * f_avg)) * reprojection_error_px
        
        # 机械臂末端的3D总误差
        gripper_error_3d_mm = np.sqrt(
            gripper_error_x_mm**2 + 
            gripper_error_y_mm**2 + 
            gripper_error_z_mm**2
        )
        
        print(f"\n步骤5: 机械臂末端位置误差")
        print(f"  末端距离: {gripper_distance_mm:.1f} mm")
        print(f"  X方向误差: σ_x = ±{gripper_error_x_mm:.3f} mm")
        print(f"  Y方向误差: σ_y = ±{gripper_error_y_mm:.3f} mm")
        print(f"  Z方向误差: σ_z = ±{gripper_error_z_mm:.3f} mm")
        print(f"  3D总误差: σ_3D = ±{gripper_error_3d_mm:.3f} mm")
    else:
        gripper_distance_mm = distance_z_mm
        gripper_error_x_mm = error_x_mm
        gripper_error_y_mm = error_y_mm
        gripper_error_z_mm = error_z_mm
        gripper_error_3d_mm = error_3d_mm
    
    # # ========================================
    # # 步骤6: 不同距离下的误差分析
    # # ========================================
    # print(f"\n步骤6: 不同工作距离下的误差预测")
    # print(f"  {'距离(mm)':<12} {'平面误差(mm)':<15} {'深度误差(mm)':<15} {'3D总误差(mm)':<15}")
    # print(f"  {'-'*60}")
    
    # test_distances = [300, 400, 500, 600, 700, 800, 900, 1000]
    # for test_z in test_distances:
    #     # 平面误差
    #     test_error_x = (test_z / fx) * reprojection_error_px
    #     test_error_y = (test_z / fy) * reprojection_error_px
    #     test_error_xy = np.sqrt(test_error_x**2 + test_error_y**2)
        
    #     # 深度误差
    #     test_error_z = (test_z**2 / (effective_baseline_mm * f_avg)) * reprojection_error_px
        
    #     # 3D总误差
    #     test_error_3d = np.sqrt(test_error_x**2 + test_error_y**2 + test_error_z**2)
        
    #     print(f"  {test_z:<12} {test_error_xy:<15.3f} {test_error_z:<15.3f} {test_error_3d:<15.3f}")
    
    # ========================================
    # 计算视场角（FOV）
    # ========================================
    # 图像传感器尺寸（根据主点估算）
    sensor_width_px = 2 * cx
    sensor_height_px = 2 * cy
    
    fov_x_rad = 2 * np.arctan(sensor_width_px / (2 * fx))
    fov_y_rad = 2 * np.arctan(sensor_height_px / (2 * fy))
    fov_x_deg = np.degrees(fov_x_rad)
    fov_y_deg = np.degrees(fov_y_rad)
    
    # ========================================
    # 返回结果
    # ========================================
    results = {
        # 输入参数
        'reprojection_error_px': reprojection_error_px,
        'distance_z_mm': distance_z_mm,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        
        # 标定板平面误差
        'error_x_mm': error_x_mm,
        'error_y_mm': error_y_mm,
        'error_xy_mm': error_xy_mm,
        'error_z_mm': error_z_mm,
        'error_z_conservative_mm': error_z_conservative_mm,
        'error_3d_mm': error_3d_mm,
        
        # 角度误差
        'angular_error_x_deg': angular_error_x_deg,
        'angular_error_y_deg': angular_error_y_deg,
        'angular_error_z_deg': angular_error_z_deg,
        'angular_error_max_deg': angular_error_max_deg,
        
        # 机械臂末端误差
        'gripper_distance_mm': gripper_distance_mm,
        'gripper_error_x_mm': gripper_error_x_mm,
        'gripper_error_y_mm': gripper_error_y_mm,
        'gripper_error_z_mm': gripper_error_z_mm,
        'gripper_error_3d_mm': gripper_error_3d_mm,
        
        # 标定板信息
        'board_width_mm': board_width_mm,
        'board_height_mm': board_height_mm,
        'board_diagonal_mm': board_diagonal_mm,
        'effective_baseline_mm': effective_baseline_mm,
        
        # 相机信息
        'fov_x_deg': fov_x_deg,
        'fov_y_deg': fov_y_deg,
    }
    
    return results


def analyze_error_sensitivity(camera_matrix, board_rows=15, board_cols=15, board_spacing_mm=65.0):
    """
    分析不同重投影误差和距离下的3D误差敏感性
    """
    print(f"\n{'='*80}")
    print(f"误差敏感性分析")
    print(f"{'='*80}")
    
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    f_avg = (fx + fy) / 2.0
    
    # 计算有效基线
    board_width_mm = (board_cols - 1) * board_spacing_mm
    board_height_mm = (board_rows - 1) * board_spacing_mm
    board_diagonal_mm = np.sqrt(board_width_mm**2 + board_height_mm**2)
    effective_baseline_mm = board_diagonal_mm / 2.0
    
    # 测试不同的重投影误差和距离
    reprojection_errors = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    distances = [400, 600, 800, 1000]
    
    print(f"\n平面位置误差 σ_xy (mm):")
    print(f"  {'重投影误差(px)':<18}", end='')
    for d in distances:
        print(f"{d}mm{' '*8}", end='')
    print()
    print(f"  {'-'*70}")
    
    for rep_err in reprojection_errors:
        print(f"  {rep_err:<18.1f}", end='')
        for d in distances:
            error_x = (d / fx) * rep_err
            error_y = (d / fy) * rep_err
            error_xy = np.sqrt(error_x**2 + error_y**2)
            print(f"{error_xy:<12.3f}", end='')
        print()
    
    print(f"\n深度误差 σ_z (mm):")
    print(f"  {'重投影误差(px)':<18}", end='')
    for d in distances:
        print(f"{d}mm{' '*8}", end='')
    print()
    print(f"  {'-'*70}")
    
    for rep_err in reprojection_errors:
        print(f"  {rep_err:<18.1f}", end='')
        for d in distances:
            error_z = (d**2 / (effective_baseline_mm * f_avg)) * rep_err
            print(f"{error_z:<12.3f}", end='')
        print()
    
    print(f"\n3D总误差 σ_3D (mm):")
    print(f"  {'重投影误差(px)':<18}", end='')
    for d in distances:
        print(f"{d}mm{' '*8}", end='')
    print()
    print(f"  {'-'*70}")
    
    for rep_err in reprojection_errors:
        print(f"  {rep_err:<18.1f}", end='')
        for d in distances:
            error_x = (d / fx) * rep_err
            error_y = (d / fy) * rep_err
            error_z = (d**2 / (effective_baseline_mm * f_avg)) * rep_err
            error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)
            print(f"{error_3d:<12.3f}", end='')
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从2D重投影误差准确计算3D空间误差")
    parser.add_argument("--reprojection-error", type=float, default=1.5,
                       help="重投影误差RMS（像素），默认1.5")
    parser.add_argument("--distance", type=float, default=600.0,
                       help="工作距离（mm），默认600")
    parser.add_argument("--camera-yaml", type=str, default="config/camera_info.yaml",
                       help="相机内参文件路径")
    parser.add_argument("--rows", type=int, default=15,
                       help="标定板行数，默认15")
    parser.add_argument("--cols", type=int, default=15,
                       help="标定板列数，默认15")
    parser.add_argument("--spacing", type=float, default=65.0,
                       help="标定板点间距（mm），默认65")
    parser.add_argument("--gripper-offset", type=float, default=0.0,
                       help="机械臂末端偏移（mm），默认0")
    parser.add_argument("--sensitivity", action="store_true",
                       help="显示误差敏感性分析")
    
    args = parser.parse_args()
    
    # 加载相机内参
    K, dist, image_size = load_camera_intrinsics(args.camera_yaml)
    
    # 计算3D误差
    results = calculate_3d_error_from_reprojection(
        reprojection_error_px=args.reprojection_error,
        camera_matrix=K,
        distance_z_mm=args.distance,
        board_rows=args.rows,
        board_cols=args.cols,
        board_spacing_mm=args.spacing,
        gripper_offset_mm=args.gripper_offset
    )
    
    # 敏感性分析
    if args.sensitivity:
        analyze_error_sensitivity(K, args.rows, args.cols, args.spacing)
    
    # ========================================
    # 总结与建议
    # ========================================
    print(f"\n{'='*80}")
    print(f"总结与建议")
    print(f"{'='*80}")
    print(f"\n对于 {args.reprojection_error:.3f} 像素的重投影误差RMS:")
    print(f"  在距离 {args.distance:.0f}mm 处:")
    print(f"    • 平面位置误差（X-Y）: ±{results['error_xy_mm']:.3f} mm")
    print(f"    • 深度误差（Z）: ±{results['error_z_mm']:.3f} mm")
    print(f"    • 3D总误差: ±{results['error_3d_mm']:.3f} mm")
    print(f"    • 最大角度误差: ±{results['angular_error_max_deg']:.4f}°")
    
    if args.gripper_offset != 0:
        print(f"\n  机械臂末端（距离 {results['gripper_distance_mm']:.0f}mm）:")
        print(f"    • 3D位置误差: ±{results['gripper_error_3d_mm']:.3f} mm")
    
    # 评估精度等级
    error_to_evaluate = results['gripper_error_3d_mm'] if args.gripper_offset != 0 else results['error_3d_mm']
    
    print(f"\n精度评估:")
    if error_to_evaluate < 0.5:
        print(f"  ✅ 优秀（< 0.5mm）- 适合高精度装配任务")
    elif error_to_evaluate < 1.0:
        print(f"  ✅ 良好（< 1mm）- 适合精密抓取和定位")
    elif error_to_evaluate < 2.0:
        print(f"  ⚠️  中等（< 2mm）- 适合一般工业操作")
    elif error_to_evaluate < 5.0:
        print(f"  ⚠️  较差（< 5mm）- 仅适合粗糙抓取")
    else:
        print(f"  ❌ 不可接受（≥ 5mm）- 需要改进")
    
    # print(f"\n改进建议:")
    # if error_to_evaluate >= 2.0:
    #     print(f"  1. 降低重投影误差:")
    #     print(f"     • 改善图像质量（照明、对比度）")
    #     print(f"     • 提高角点检测精度")
    #     print(f"     • 使用亚像素精度算法")
    #     print(f"  2. 优化工作距离:")
    #     print(f"     • 当前距离: {args.distance:.0f}mm")
    #     print(f"     • 建议距离: 400-600mm（对于此配置）")
    #     print(f"  3. 改善标定配置:")
    #     print(f"     • 使用更大的标定板（当前: {results['board_diagonal_mm']:.1f}mm对角线）")
    #     print(f"     • 增加标定姿态的多样性")
    #     print(f"     • 确保标定板覆盖整个工作区域")
    # else:
    #     print(f"  当前配置已达到良好精度，继续保持:")
    #     print(f"     • 重投影误差 < 1.0 像素")
    #     print(f"     • 工作距离在合理范围内")
    #     print(f"     • 标定板配置适当")
    
    # # 理论精度极限
    # print(f"\n理论精度极限（重投影误差=0.1px）:")
    # theoretical_results = calculate_3d_error_from_reprojection(
    #     reprojection_error_px=0.1,
    #     camera_matrix=K,
    #     distance_z_mm=args.distance,
    #     board_rows=args.rows,
    #     board_cols=args.cols,
    #     board_spacing_mm=args.spacing,
    #     gripper_offset_mm=args.gripper_offset
    # )
    # print(f"  在距离 {args.distance:.0f}mm 处，理论最佳3D误差: ±{theoretical_results['error_3d_mm']:.3f} mm")
    # print(f"  当前误差是理论极限的 {error_to_evaluate/theoretical_results['error_3d_mm']:.1f} 倍")
