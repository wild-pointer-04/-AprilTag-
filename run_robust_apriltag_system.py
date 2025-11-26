#!/usr/bin/env python3
"""
运行鲁棒AprilTag系统 - 解决247像素重投影误差问题

这是集成了PnP多解歧义解决方案的主程序
"""

import cv2
import numpy as np
import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.robust_apriltag_system import RobustAprilTagSystem
from src.utils import load_camera_intrinsics
from src.detect_grid_improved import try_find_adaptive
from src.utils import rvec_to_camera_tilt

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def process_single_image(image_path: str,
                        robust_system: RobustAprilTagSystem,
                        camera_matrix: np.ndarray,
                        dist_coeffs: np.ndarray,
                        grid_rows: int = 15,
                        grid_cols: int = 15,
                        save_visualization: bool = True) -> dict:
    """
    处理单张图像
    
    Args:
        image_path: 图像路径
        robust_system: 鲁棒AprilTag系统
        camera_matrix: 相机内参
        dist_coeffs: 畸变系数
        grid_rows: 网格行数
        grid_cols: 网格列数
        save_visualization: 是否保存可视化结果
        
    Returns:
        处理结果字典
    """
    print(f"\n🔍 处理图像: {image_path}")
    print("-" * 60)
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"无法加载图像: {image_path}")
        return {'success': False, 'error': 'Failed to load image'}
    
    # 畸变矫正
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    
    # 检测标定板角点
    print("📍 检测标定板角点...")
    ret, corners, keypoints = try_find_adaptive(gray, grid_rows, grid_cols)
    
    if not ret or corners is None:
        logger.warning("未检测到标定板角点")
        return {'success': False, 'error': 'Grid detection failed'}
    
    corners = corners.reshape(-1, 2)
    print(f"✅ 检测到 {len(corners)} 个角点")
    
    # 鲁棒位姿估计
    print("🎯 执行鲁棒位姿估计...")
    success, rvec, tvec, error, info = robust_system.robust_pose_estimation(
        undistorted, corners, camera_matrix, dist_coeffs, grid_rows, grid_cols
    )
    
    result = {
        'image_path': image_path,
        'success': success,
        'corners_detected': len(corners),
        'reprojection_error': error
    }
    
    if success and error < robust_system.pnp_resolver.max_reprojection_error:
        # 计算倾斜角度
        roll, pitch, yaw = rvec_to_camera_tilt(rvec)
        
        result.update({
            'roll_deg': roll,
            'pitch_deg': pitch, 
            'yaw_deg': yaw,
            'pnp_method': info['pnp_info'].get('method', 'Unknown'),
            'apriltag_consistent': info['pnp_info'].get('apriltag_consistency', {}).get('is_consistent', False),
            'apriltag_info': info.get('apriltag_info', {})
        })
        
        print(f"✅ 位姿估计成功!")
        print(f"  重投影误差: {error:.3f}px")
        print(f"  使用方法: {result['pnp_method']}")
        print(f"  AprilTag一致性: {'✅' if result['apriltag_consistent'] else '⚠️'}")
        print(f"  相机倾斜角度:")
        print(f"    Roll:  {roll:+7.2f}°")
        print(f"    Pitch: {pitch:+7.2f}°") 
        print(f"    Yaw:   {yaw:+7.2f}°")
        
        # 可视化结果
        if save_visualization:
            vis_image = visualize_results(undistorted, corners, info, rvec, tvec, 
                                        camera_matrix, dist_coeffs)
            
            # 保存可视化图像
            output_dir = Path('outputs/robust_apriltag_results')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            vis_path = output_dir / f'{image_name}_robust_result.png'
            cv2.imwrite(str(vis_path), vis_image)
            result['visualization_path'] = str(vis_path)
            print(f"  可视化结果: {vis_path}")
        
    else:
        print(f"❌ 位姿估计失败或误差过大")
        print(f"  重投影误差: {error:.3f}px")
        if 'pnp_info' in info:
            print(f"  尝试的方法数: {info['pnp_info'].get('total_solutions', 0)}")
        
        result['error'] = 'High reprojection error or pose estimation failed'
    
    return result


def visualize_results(image: np.ndarray,
                     corners: np.ndarray,
                     info: dict,
                     rvec: np.ndarray,
                     tvec: np.ndarray,
                     camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray) -> np.ndarray:
    """可视化结果"""
    
    vis_image = image.copy()
    
    # 绘制检测到的角点
    for corner in corners:
        cv2.circle(vis_image, tuple(corner.astype(int)), 3, (0, 255, 0), -1)
    
    # 绘制AprilTag信息
    if 'apriltag_info' in info:
        apriltag_info = info['apriltag_info']
        
        # 绘制AprilTag边框
        if 'tag_corners' in apriltag_info:
            tag_corners = apriltag_info['tag_corners'].astype(int)
            cv2.polylines(vis_image, [tag_corners], True, (0, 255, 255), 3)
        
        # 绘制AprilTag中心和ID
        if 'tag_center' in apriltag_info:
            center = apriltag_info['tag_center'].astype(int)
            cv2.circle(vis_image, tuple(center), 8, (0, 255, 255), -1)
            
            if 'tag_id' in apriltag_info:
                cv2.putText(vis_image, f"ID:{apriltag_info['tag_id']}", 
                           (center[0]-20, center[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制坐标系
        if 'origin_2d' in apriltag_info:
            origin = apriltag_info['origin_2d'].astype(int)
            cv2.circle(vis_image, tuple(origin), 6, (0, 0, 255), -1)
            
            # 绘制坐标轴
            if 'x_direction_2d' in apriltag_info and 'y_direction_2d' in apriltag_info:
                axis_length = 50
                x_end = origin + (apriltag_info['x_direction_2d'] * axis_length).astype(int)
                y_end = origin + (apriltag_info['y_direction_2d'] * axis_length).astype(int)
                
                cv2.arrowedLine(vis_image, tuple(origin), tuple(x_end), (0, 0, 255), 2)
                cv2.arrowedLine(vis_image, tuple(origin), tuple(y_end), (0, 255, 0), 2)
                
                cv2.putText(vis_image, "X", tuple(x_end + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(vis_image, "Y", tuple(y_end + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 添加信息文本
    info_text = []
    if 'pnp_info' in info:
        pnp_info = info['pnp_info']
        info_text.append(f"Method: {pnp_info.get('method', 'Unknown')}")
        
        if 'apriltag_consistency' in pnp_info:
            consistency = pnp_info['apriltag_consistency']
            if 'angle_difference_deg' in consistency:
                info_text.append(f"Angle diff: {consistency['angle_difference_deg']:.1f}deg")
    
    # 绘制信息文本
    y_offset = 30
    for i, text in enumerate(info_text):
        cv2.putText(vis_image, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_image


def process_multiple_images(data_dir: str,
                           robust_system: RobustAprilTagSystem,
                           camera_matrix: np.ndarray,
                           dist_coeffs: np.ndarray,
                           grid_rows: int = 4,
                           grid_cols: int = 11) -> list:
    """处理多张图像"""
    
    # 查找图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(data_dir).glob(f'*{ext}'))
        image_paths.extend(Path(data_dir).glob(f'*{ext.upper()}'))
    
    image_paths = sorted([str(p) for p in image_paths])
    
    if not image_paths:
        logger.error(f"在目录 {data_dir} 中未找到图像文件")
        return []
    
    print(f"\n🚀 开始处理 {len(image_paths)} 张图像")
    print("=" * 80)
    
    results = []
    successful_count = 0
    total_error = 0.0
    max_error = 0.0
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}]", end=" ")
        
        result = process_single_image(
            image_path, robust_system, camera_matrix, dist_coeffs,
            grid_rows, grid_cols, save_visualization=True
        )
        
        results.append(result)
        
        if result['success'] and result['reprojection_error'] < robust_system.pnp_resolver.max_reprojection_error:
            successful_count += 1
            total_error += result['reprojection_error']
            max_error = max(max_error, result['reprojection_error'])
    
    # 生成统计报告
    print("\n" + "=" * 80)
    print("🎯 鲁棒AprilTag系统处理结果统计")
    print("=" * 80)
    
    print(f"总图像数: {len(image_paths)}")
    print(f"成功处理: {successful_count}")
    print(f"成功率: {successful_count/len(image_paths)*100:.1f}%")
    
    if successful_count > 0:
        avg_error = total_error / successful_count
        print(f"平均重投影误差: {avg_error:.3f}px ✅")
        print(f"最大重投影误差: {max_error:.3f}px ✅")
        
        # 统计使用的方法
        methods = {}
        consistent_count = 0
        
        for result in results:
            if result['success'] and 'pnp_method' in result:
                method = result['pnp_method']
                methods[method] = methods.get(method, 0) + 1
                
                if result.get('apriltag_consistent', False):
                    consistent_count += 1
        
        print(f"AprilTag一致性: {consistent_count}/{successful_count} ({consistent_count/successful_count*100:.1f}%)")
        
        print(f"\n使用的PnP方法统计:")
        for method, count in methods.items():
            print(f"  {method}: {count} 次 ({count/successful_count*100:.1f}%)")
    
    # 保存详细结果
    save_detailed_results(results)
    
    return results


def save_detailed_results(results: list):
    """保存详细结果到CSV文件"""
    
    output_dir = Path('outputs/robust_apriltag_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / 'detailed_results.csv'
    
    with open(csv_path, 'w') as f:
        # 写入表头
        f.write('image_path,success,reprojection_error,roll_deg,pitch_deg,yaw_deg,')
        f.write('pnp_method,apriltag_consistent,corners_detected\n')
        
        # 写入数据
        for result in results:
            f.write(f"{result.get('image_path', 'N/A')},{result.get('success', False)},")
            f.write(f"{result.get('reprojection_error', 'inf')},")
            f.write(f"{result.get('roll_deg', 'N/A')},")
            f.write(f"{result.get('pitch_deg', 'N/A')},")
            f.write(f"{result.get('yaw_deg', 'N/A')},")
            f.write(f"{result.get('pnp_method', 'N/A')},")
            f.write(f"{result.get('apriltag_consistent', False)},")
            f.write(f"{result.get('corners_detected', 0)}\n")
    
    print(f"\n📊 详细结果已保存: {csv_path}")


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='鲁棒AprilTag系统 - 解决PnP多解歧义问题')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='图像数据目录 (默认: data)')
    parser.add_argument('--image', type=str, 
                       help='单张图像路径 (如果指定，则只处理这张图像)')
    parser.add_argument('--camera-yaml', type=str, default='config/camera_info.yaml',
                       help='相机内参文件 (默认: config/camera_info.yaml)')
    parser.add_argument('--tag-family', type=str, default='tagStandard41h12',
                       help='AprilTag家族 (默认: tagStandard41h12)')
    parser.add_argument('--tag-size', type=float, default=20.0,
                       help='AprilTag尺寸(mm) (默认: 20.0)')
    parser.add_argument('--grid-rows', type=int, default=4,
                       help='标定板网格行数 (默认: 4)')
    parser.add_argument('--grid-cols', type=int, default=11,
                       help='标定板网格列数 (默认: 11)')
    parser.add_argument('--max-error', type=float, default=10.0,
                       help='最大允许重投影误差(px) (默认: 10.0)')
    
    args = parser.parse_args()
    
    print("🔧 鲁棒AprilTag系统启动")
    print("专门解决247像素重投影误差问题")
    print("=" * 80)
    
    # 加载相机参数
    print("📷 加载相机参数...")
    try:
        result = load_camera_intrinsics(args.camera_yaml)
        if len(result) == 3:
            camera_matrix, dist_coeffs, image_size = result
        else:
            camera_matrix, dist_coeffs = result
        
        if camera_matrix is None:
            logger.error("无法加载相机参数")
            return
            
        print("✅ 相机参数加载成功")
        
    except Exception as e:
        logger.error(f"加载相机参数失败: {e}")
        return
    
    # 初始化鲁棒AprilTag系统
    print(f"🎯 初始化鲁棒AprilTag系统...")
    print(f"  AprilTag家族: {args.tag_family}")
    print(f"  AprilTag尺寸: {args.tag_size}mm")
    print(f"  网格尺寸: {args.grid_rows}x{args.grid_cols}")
    print(f"  最大误差阈值: {args.max_error}px")
    
    robust_system = RobustAprilTagSystem(
        tag_family=args.tag_family,
        tag_size=args.tag_size,
        max_reprojection_error=args.max_error
    )
    
    # 处理图像
    if args.image:
        # 处理单张图像
        result = process_single_image(
            args.image, robust_system, camera_matrix, dist_coeffs,
            args.grid_rows, args.grid_cols, save_visualization=True
        )
        
        if result['success']:
            print(f"\n🎉 单张图像处理成功!")
        else:
            print(f"\n❌ 单张图像处理失败: {result.get('error', 'Unknown error')}")
    
    else:
        # 处理多张图像
        results = process_multiple_images(
            args.data_dir, robust_system, camera_matrix, dist_coeffs,
            args.grid_rows, args.grid_cols
        )
        
        successful_results = [r for r in results if r['success'] and 
                            r['reprojection_error'] < args.max_error]
        
        print(f"\n🎉 批量处理完成!")
        print(f"成功处理 {len(successful_results)}/{len(results)} 张图像")
        
        if successful_results:
            avg_error = np.mean([r['reprojection_error'] for r in successful_results])
            print(f"平均重投影误差: {avg_error:.3f}px (目标: <{args.max_error}px)")
            
            if avg_error < 10.0:
                print("✅ 成功解决PnP多解歧义问题!")
            else:
                print("⚠️ 仍有改进空间，建议调整参数")


if __name__ == '__main__':
    main()