import os
import sys
import argparse
import glob
import cv2

# 添加项目根目录到 Python 路径，以便导入 src 模块
# 无论从项目根目录还是 src 目录运行都能正常工作
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.estimate_tilt import solve_pose_with_guess, visualize_and_save

def process_one(image_path: str, save_path: str, rows: int | None, cols: int | None, asymmetric: bool, spacing: float, auto: bool, 
                auto_search_func, try_find_func, refine_func, camera_yaml_path: str = None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) 检测圆点
    blob_keypoints = None  # 用于存储 blob 检测点
    # 如果指定了 --auto，则自动搜索；否则使用指定的 rows/cols（默认15x15）
    if auto:
        result = auto_search_func(gray, rows_range=(8, 26), cols_range=(8, 32))  # 扩大搜索范围，包含8×8
        if len(result) == 4:
            ok, corners, meta, blob_keypoints = result
        else:
            ok, corners, meta = result
            # 如果没有返回 keypoints，单独检测一次
            if blob_keypoints is None:
                from src.utils import build_blob_detector
                det = build_blob_detector()
                blob_keypoints = det.detect(gray)
        if not ok:
            raise RuntimeError("自动搜索失败：未检测到可用圆点网格。可尝试指定 --rows --cols。")
        rows, cols, symmetric = meta
        print(f"[AUTO] 检测到网格 rows={rows}, cols={cols}, symmetric={symmetric}")
    else:
        symmetric = not asymmetric
        result = try_find_func(gray, rows, cols, symmetric=symmetric)
        if len(result) == 3:
            ok, centers, blob_keypoints = result
        else:
            ok, centers = result
            # 如果没有返回 keypoints，单独检测一次
            if blob_keypoints is None:
                from src.utils import build_blob_detector
                det = build_blob_detector()
                blob_keypoints = det.detect(gray)
        if not ok:
            print("[WARN] 按给定 rows/cols 未找到网格，自动回退到 auto 搜索……")
            result = auto_search_func(gray, rows_range=(8, 26), cols_range=(8, 32))  # 扩大搜索范围
            if len(result) == 4:
                ok, corners_auto, meta, blob_keypoints = result
            else:
                ok, corners_auto, meta = result
                if blob_keypoints is None:
                    from src.utils import build_blob_detector
                    det = build_blob_detector()
                    blob_keypoints = det.detect(gray)
            if not ok:
                raise RuntimeError("按给定 rows/cols 未找到网格，且自动搜索也失败。请检查参数或图像质量。")
            rows, cols, symmetric = meta
            print(f"[AUTO-FALLBACK] 检测到网格 rows={rows}, cols={cols}, symmetric={symmetric}")
            corners = corners_auto
        else:
            corners = refine_func(gray, centers)

    # 2) PnP 求位姿（内部根据对称性尝试多种排列）
    rvec, tvec, angles_dict, K, dist, ordered_corners = solve_pose_with_guess(
        gray, corners, rows, cols, spacing=spacing, symmetric=symmetric, camera_yaml_path=camera_yaml_path
    )
    
    pts2d = ordered_corners.reshape(-1, 2)
    center_mean = pts2d.mean(axis=0)
    center_idx = (rows // 2) * cols + (cols // 2)
    center_mid = pts2d[center_idx]
    
    print("\n== 板子中心 (像素坐标) ==")
    print(f"均值中心: (u,v)=({center_mean[0]:.1f}, {center_mean[1]:.1f})")
    print(f"中点中心: (u,v)=({center_mid[0]:.1f}, {center_mid[1]:.1f})  (rows={rows}, cols={cols})")
    
    roll_euler, pitch_euler, yaw_euler = angles_dict['euler']
    roll_tilt, pitch_tilt, yaw_tilt = angles_dict['camera_tilt']
    symmetry_info = angles_dict.get('symmetry', {})
    if symmetry_info:
        print(f"对称解选择: {symmetry_info.get('transform')} (平均重投影误差={symmetry_info.get('reprojection_error', 0):.4f}px)")

    # 4) 输出判定与可视化
    print("\n== 标准欧拉角（板子相对于相机）XYZ顺序 ==")
    print(f"Roll(前后仰): {roll_euler:+.3f}°")
    print(f"Pitch(平面旋): {pitch_euler:+.3f}°")
    print(f"Yaw  (左右歪): {yaw_euler:+.3f}°")
    
    print("\n== 相机倾斜角（假设板子水平，相机相对于水平面）==")
    print(f"Roll(前后仰): {roll_tilt:+.3f}°")
    print(f"Pitch(平面旋): {pitch_tilt:+.3f}°")
    print(f"Yaw  (左右歪): {yaw_tilt:+.3f}°")
    print("\n[注意] 相机倾斜角假设板子是水平放置的。如果板子本身倾斜，结果可能不准确。")

    # 使用相机倾斜角判断（如果板子水平，这个更准确）
    tol = 0.5
    tilt = (abs(roll_tilt) > tol) or (abs(pitch_tilt) > tol)
    print(f"\n歪斜判断（阈值±{tol}°）：{'存在歪斜' if tilt else '基本无歪斜'}")
    if tilt:
        print(f"   -> Roll偏移: {roll_tilt:+.3f}°")
        print(f"   -> Pitch偏移: {pitch_tilt:+.3f}°")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存原结果图（不带blob点）
    out_path = visualize_and_save(
        img, ordered_corners, K, dist, rvec, tvec, save_path,
        center_px=center_mid,
        center_mean_px=center_mean,
        blob_keypoints=None,  # 不显示blob点
    )
    print(f"\n结果图已保存：{out_path}")
    
    # 保存带blob点的图片（新文件，不覆盖原图）
    if blob_keypoints is not None:
        # 生成新文件名：原文件名 + "_with_blobs"
        base_path = os.path.splitext(save_path)[0]  # 去掉扩展名
        ext = os.path.splitext(save_path)[1]  # 获取扩展名
        blob_save_path = f"{base_path}_with_blobs{ext}"
        
        out_path_blob = visualize_and_save(
            img, ordered_corners, K, dist, rvec, tvec, blob_save_path,
            center_px=center_mid,
            center_mean_px=center_mean,
            blob_keypoints=blob_keypoints,  # 显示blob点
        )
        print(f"带blob点的结果图已保存：{out_path_blob}")


def main():
    parser = argparse.ArgumentParser(description="Dot-board tilt checker (OpenCV + PnP)")
    parser.add_argument("--image", help="输入图像路径")
    parser.add_argument("--dir", help="输入目录（批量处理）")
    parser.add_argument("--pattern", default="*.png,*.jpg,*.jpeg", help="批量处理的通配符，逗号分隔")
    parser.add_argument("--rows", type=int, default=15, help="圆点行数(内点)，默认15")
    parser.add_argument("--cols", type=int, default=15, help="圆点列数(内点)，默认15")
    parser.add_argument("--asymmetric", action="store_true", help="非对称圆点网格")
    parser.add_argument("--spacing", type=float, default=10.0, help="相邻圆点间距(mm/任意单位)")
    parser.add_argument("--auto", action="store_true", help="自动搜索 rows/cols/对称性（如果指定此选项，会忽略 --rows 和 --cols）")
    parser.add_argument("--save", default="outputs/result.png", help="结果图保存路径")
    parser.add_argument("--use-original", action="store_true", help="使用原版检测算法（默认使用改进版）")
    parser.add_argument("--camera-yaml", type=str, default="config/camera_info.yaml", 
                       help="相机内参 YAML 文件路径（如果存在则使用真实内参，否则使用默认近似值）")
    args = parser.parse_args()
    
    # 根据参数选择使用原版还是改进版
    if args.use_original:
        from src.detect_grid import auto_search, try_find, refine
        print("[INFO] 使用原版检测算法")
    else:
        try:
            from src.detect_grid_improved import auto_search, try_find, refine
            print("[INFO] 使用改进版检测算法（自适应参数+智能搜索）")
        except ImportError:
            from src.detect_grid import auto_search, try_find, refine
            print("[WARN] 改进版不可用，回退到原版检测算法")

    if args.dir:
        patterns = [p.strip() for p in args.pattern.split(',') if p.strip()]
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(args.dir, pat)))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"目录为空或无匹配文件：{args.dir} ({args.pattern})")
        for fp in files:
            name = os.path.splitext(os.path.basename(fp))[0]
            save_path = os.path.join("outputs", f"{name}_result.png")
            print("\n==============================")
            print(f"处理图片：{fp}")
            process_one(fp, save_path, args.rows, args.cols, args.asymmetric, args.spacing, args.auto,
                       auto_search, try_find, refine, camera_yaml_path=args.camera_yaml)
    else:
        if not args.image:
            raise ValueError("请提供 --image 或 --dir")
        process_one(args.image, args.save, args.rows, args.cols, args.asymmetric, args.spacing, args.auto,
                   auto_search, try_find, refine, camera_yaml_path=args.camera_yaml)

if __name__ == "__main__":
    main()
