#!/usr/bin/env python3
"""
像素角分辨率（PPD / DPP）计算工具。

特性：
- 自动根据输入来源（rosbag 或 图像目录）读取首个可用画面并解析图像尺寸
- 基于相机内参计算横/纵/对角方向的 Pixels Per Degree (PPD) 与 Degrees Per Pixel (DPP)
- 如果 YAML 中记录的图像尺寸与实际帧尺寸不同，会自动缩放内参矩阵

示例：
    # 从 rosbag 读取
    python compute_pixel_angular_resolution.py \
        --camera-yaml config/camera_info.yaml \
        --rosbag rosbags/testbag \
        --image-topic /left/color/image_raw
    
    # 从数据集图片读取
    python compute_pixel_angular_resolution.py \
        --camera-yaml config/camera_info.yaml \
        --image-dir data/images
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Generator, Iterable, Optional, Tuple

import cv2
import numpy as np

from src.utils import load_camera_intrinsics, scale_camera_intrinsics, get_camera_intrinsics


def _iter_image_files(image_dir: Path) -> Generator[Tuple[np.ndarray, str], None, None]:
    """按名称排序遍历目录下的所有图像文件。"""
    if not image_dir.exists():
        raise FileNotFoundError(f'图像目录不存在: {image_dir}')
    
    image_paths = sorted([
        p for p in image_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    ])
    
    if not image_paths:
        raise RuntimeError(f'在目录 {image_dir} 中未找到任何图像文件')
    
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f'无法读取图像: {path}')
            continue
        yield img, str(path)


def _iter_rosbag_frames(
    bag_path: Path,
    image_topic: str,
    max_frames: Optional[int] = 1
) -> Generator[Tuple[np.ndarray, str], None, None]:
    """从 rosbag2 (SQLite) 中遍历图像帧。"""
    try:
        from cv_bridge import CvBridge
        from rclpy.serialization import deserialize_message
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        from sensor_msgs.msg import Image as RosImage
    except ImportError as exc:
        raise RuntimeError(
            '读取 rosbag 需要 ros-humble-rosbag2-py、rosidl 以及 cv_bridge，请先安装相关依赖'
        ) from exc
    
    if not bag_path.exists():
        raise FileNotFoundError(f'rosbag 路径不存在: {bag_path}')
    
    bridge = CvBridge()
    storage_options = StorageOptions(uri=str(bag_path), storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    
    if image_topic not in topic_types:
        raise RuntimeError(
            f'在 rosbag 中未找到话题 {image_topic}，可用话题: {list(topic_types)}'
        )
    
    frames_yielded = 0
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic != image_topic:
            continue
        
        msg = deserialize_message(data, RosImage)
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        yield cv_image, f'{bag_path.name}:{timestamp}'
        
        frames_yielded += 1
        if max_frames is not None and frames_yielded >= max_frames:
            break


def _prepare_intrinsics(
    camera_yaml: Path,
    target_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """加载并针对指定图像尺寸准备相机内参。"""
    try:
        result = load_camera_intrinsics(str(camera_yaml))
        if len(result) == 3:
            K, dist, yaml_size = result
        else:
            K, dist = result
            yaml_size = None
    except Exception as exc:
        print(f'无法从 {camera_yaml} 读取内参，原因: {exc}，改用默认针孔模型')
        height, width = target_size[1], target_size[0]
        K, dist = get_camera_intrinsics(height, width, yaml_path=None, f_scale=1.0)
        return K, dist
    
    if K is None:
        height, width = target_size[1], target_size[0]
        K, dist = get_camera_intrinsics(height, width, yaml_path=None, f_scale=1.0)
        return K, dist
    
    if yaml_size is not None and (yaml_size[0] != target_size[0] or yaml_size[1] != target_size[1]):
        K_scaled, dist_scaled = scale_camera_intrinsics(
            K, dist, yaml_size, target_size
        )
        print(
            f'自动缩放内参矩阵: YAML 尺寸 {yaml_size} -> 实际尺寸 {target_size}, '
            f'缩放比 ({target_size[0]/yaml_size[0]:.3f}, {target_size[1]/yaml_size[1]:.3f})'
        )
        return K_scaled, dist_scaled
    
    return K.copy(), dist.copy()


def _compute_ppd_stats(
    K: np.ndarray,
    image_size: Tuple[int, int]
) -> Dict[str, float]:
    """
    计算中心PPD（标准定义）与平均PPD，并附带FOV/各向异性等诊断信息。
    
    中心PPD公式推导：
    - 在针孔相机模型中，像素坐标 u = fx * tan(θ) + cx（θ为视角）
    - 视角变化 Δθ 时，像素变化 Δu = fx * (tan(θ+Δθ) - tan(θ))
    - 在图像中心附近（θ≈0），tan(θ+Δθ) - tan(θ) ≈ Δθ（小角度近似）
    - 因此，PPD = fx * (π/180) ≈ fx * 0.017453（1° = π/180 弧度）
    """
    width, height = image_size
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    
    # 视场角（供参考）
    hfov_rad = 2.0 * math.atan(width / (2.0 * fx))
    vfov_rad = 2.0 * math.atan(height / (2.0 * fy))
    hfov_deg = math.degrees(hfov_rad)
    vfov_deg = math.degrees(vfov_rad)
    
    # 中心 PPD: 视角变化 1° 时的像素位移（推荐指标）
    # 正确公式：在图像中心，视角变化 Δθ 弧度时，像素变化 Δu = fx * Δθ
    # 当 Δθ = 1° = π/180 弧度时，PPD = fx * (π/180)
    deg_to_rad = math.pi / 180.0
    ppd_x_center = fx * deg_to_rad
    ppd_y_center = fy * deg_to_rad
    ppd_avg_center = (ppd_x_center + ppd_y_center) / 2.0
    dpp_x_center = 1.0 / ppd_x_center
    dpp_y_center = 1.0 / ppd_y_center
    dpp_avg_center = (dpp_x_center + dpp_y_center) / 2.0
    
    # 平均 PPD（整幅图像宽/高除以 FOV）
    ppd_x_average = width / hfov_deg
    ppd_y_average = height / vfov_deg
    ppd_avg_average = (ppd_x_average + ppd_y_average) / 2.0
    dpp_x_average = 1.0 / ppd_x_average
    dpp_y_average = 1.0 / ppd_y_average
    dpp_avg_average = (dpp_x_average + dpp_y_average) / 2.0
    
    # 对角线 PPD（供参考）
    tan_h = math.tan(hfov_rad / 2.0)
    tan_v = math.tan(vfov_rad / 2.0)
    diag_half = math.sqrt(tan_h ** 2 + tan_v ** 2)
    diag_fov_rad = 2.0 * math.atan(diag_half)
    diag_fov_deg = math.degrees(diag_fov_rad)
    diag_pixels = math.hypot(width, height)
    ppd_diag_average = diag_pixels / diag_fov_deg
    dpp_diag_average = 1.0 / ppd_diag_average
    
    anisotropy_ratio = ppd_x_center / ppd_y_center if ppd_y_center != 0 else float('inf')
    anisotropy_percent = abs(1.0 - anisotropy_ratio) * 100.0
    
    return {
        'width_px': width,
        'height_px': height,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'hfov_deg': hfov_deg,
        'vfov_deg': vfov_deg,
        'diag_fov_deg': diag_fov_deg,
        'ppd_horizontal_center': ppd_x_center,
        'ppd_vertical_center': ppd_y_center,
        'ppd_average_center': ppd_avg_center,
        'ppd_horizontal_average': ppd_x_average,
        'ppd_vertical_average': ppd_y_average,
        'ppd_diagonal_average': ppd_diag_average,
        'ppd_average_average': ppd_avg_average,
        'dpp_horizontal_center': dpp_x_center,
        'dpp_vertical_center': dpp_y_center,
        'dpp_average_center': dpp_avg_center,
        'dpp_horizontal_average': dpp_x_average,
        'dpp_vertical_average': dpp_y_average,
        'dpp_diagonal_average': dpp_diag_average,
        'dpp_average_average': dpp_avg_average,
        'anisotropy_ratio': anisotropy_ratio,
        'anisotropy_percent': anisotropy_percent
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='计算相机像素角分辨率 (PPD/DPP)'
    )
    parser.add_argument('--camera-yaml', type=str, required=True,
                        help='相机内参 YAML 文件路径')
    parser.add_argument('--rosbag', type=str, default=None,
                        help='rosbag2 目录路径（提供则从 rosbag 读取）')
    parser.add_argument('--image-topic', type=str, default='/camera/color/image_raw',
                        help='rosbag 中的图像话题名称')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='图像目录（当未提供 --rosbag 时使用）')
    parser.add_argument('--max-frames', type=int, default=1,
                        help='为校验尺寸最多读取的帧数')
    parser.add_argument('--save-json', type=str, default=None,
                        help='可选，保存结果到 JSON 文件')
    return parser.parse_args()


def main():
    args = parse_args()
    source_desc = ''
    
    if args.rosbag:
        frame_iter = _iter_rosbag_frames(
            Path(args.rosbag),
            args.image_topic,
            max_frames=args.max_frames
        )
        source_desc = f'rosbag {args.rosbag} ({args.image_topic})'
    else:
        image_dir = Path(args.image_dir) if args.image_dir else Path('data')
        frame_iter = _iter_image_files(image_dir)
        source_desc = f'图像目录 {image_dir}'
    
    frame_sizes = []
    for idx, (frame, frame_id) in enumerate(frame_iter):
        height, width = frame.shape[:2]
        frame_sizes.append((width, height))
        print(f'读取帧 {idx+1}: {frame_id} ({width}x{height})')
        if idx + 1 >= args.max_frames:
            break
    
    if not frame_sizes:
        raise RuntimeError('未能读取任何图像帧，无法计算 PPD')
    
    unique_sizes = sorted(set(frame_sizes))
    results = []
    camera_yaml_path = Path(args.camera_yaml)
    
    for size in unique_sizes:
        print(f'计算尺寸 {size[0]}x{size[1]} 的像素角分辨率...')
        K, _ = _prepare_intrinsics(camera_yaml_path, size)
        metrics = _compute_ppd_stats(K, size)
        metrics['source'] = source_desc
        results.append(metrics)
    
    print('\n====== 像素角分辨率结果 ======')
    for metrics in results:
        print(f"来源: {metrics['source']}")
        print(f"图像尺寸: {metrics['width_px']} x {metrics['height_px']} px")
        print(f"水平 FOV: {metrics['hfov_deg']:.3f}°")
        print(f"垂直 FOV: {metrics['vfov_deg']:.3f}°")
        print(f"对角 FOV: {metrics['diag_fov_deg']:.3f}°")
        print("中心 PPD (推荐): "
              f"H {metrics['ppd_horizontal_center']:.2f} / "
              f"V {metrics['ppd_vertical_center']:.2f} / "
              f"Avg {metrics['ppd_average_center']:.2f} px/°")
        print("中心 DPP: "
              f"H {metrics['dpp_horizontal_center']:.5f} / "
              f"V {metrics['dpp_vertical_center']:.5f} / "
              f"Avg {metrics['dpp_average_center']:.5f} °/px")
        print("平均 PPD (整幅 FOV): "
              f"H {metrics['ppd_horizontal_average']:.2f} / "
              f"V {metrics['ppd_vertical_average']:.2f} / "
              f"D {metrics['ppd_diagonal_average']:.2f} / "
              f"Avg {metrics['ppd_average_average']:.2f} px/°")
        print("平均 DPP: "
              f"H {metrics['dpp_horizontal_average']:.5f} / "
              f"V {metrics['dpp_vertical_average']:.5f} / "
              f"D {metrics['dpp_diagonal_average']:.5f} / "
              f"Avg {metrics['dpp_average_average']:.5f} °/px")
        print(f"各向异性: ratio={metrics['anisotropy_ratio']:.4f}, "
              f"Δ={metrics['anisotropy_percent']:.3f}%")
        print('-' * 40)
    
    if args.save_json:
        import json
        json_path = Path(args.save_json)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'💾 已保存结果到 {json_path}')


if __name__ == '__main__':
    main()

