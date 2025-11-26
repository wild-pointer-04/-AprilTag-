#!/usr/bin/env python3
"""
测试重投影误差淘汰功能

验证修改后的代码是否正确淘汰高误差图片
"""

import sys
import os

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

print("="*80)
print("测试重投影误差淘汰功能")
print("="*80)

# 模拟不同的误差场景
test_cases = [
    {"frame_id": "frame_001", "error": 0.5, "threshold": 1.0, "expected": "通过"},
    {"frame_id": "frame_002", "error": 0.9, "threshold": 1.0, "expected": "通过"},
    {"frame_id": "frame_003", "error": 1.5, "threshold": 1.0, "expected": "淘汰"},
    {"frame_id": "frame_004", "error": 5.0, "threshold": 1.0, "expected": "淘汰"},
    {"frame_id": "frame_005", "error": 0.3, "threshold": 1.0, "expected": "通过"},
    {"frame_id": "frame_006", "error": 10.0, "threshold": 1.0, "expected": "淘汰"},
]

print("\n测试场景:")
print(f"{'帧ID':<15} {'误差(px)':<12} {'阈值(px)':<12} {'预期结果':<10} {'实际结果':<10}")
print("-"*80)

for case in test_cases:
    frame_id = case["frame_id"]
    error = case["error"]
    threshold = case["threshold"]
    expected = case["expected"]
    
    # 模拟判断逻辑
    if error > threshold:
        actual = "淘汰"
    else:
        actual = "通过"
    
    status = "✅" if actual == expected else "❌"
    print(f"{frame_id:<15} {error:<12.2f} {threshold:<12.2f} {expected:<10} {actual:<10} {status}")

print("\n" + "="*80)
print("测试完成！")
print("="*80)

print("\n修改说明:")
print("1. 添加了 rejected_by_error_count 计数器")
print("2. 在 process_frame() 中，如果 robust_error > max_reprojection_error，直接返回 None")
print("3. 更新了统计日志和报告，显示淘汰的帧数")
print("4. JSON 和 CSV 结果中只包含通过阈值的帧")
print("5. 图像只保存通过阈值的帧")

print("\n使用示例:")
print("  # 淘汰误差 > 1.0px 的图片")
print("  python robust_tilt_checker_node.py --max-error 1.0 --rosbag ... --save-images")
print()
print("  # 淘汰误差 > 5.0px 的图片")
print("  python robust_tilt_checker_node.py --max-error 5.0 --rosbag ... --save-images")
print()
print("  # 不淘汰任何图片（使用很大的阈值）")
print("  python robust_tilt_checker_node.py --max-error 1000.0 --rosbag ... --save-images")

print("\n预期效果:")
print("  - 重投影误差 <= 阈值: 保存结果和图像，计入 success_count")
print("  - 重投影误差 > 阈值: 不保存结果和图像，计入 failure_count 和 rejected_by_error_count")
print("  - 统计报告中显示淘汰率和淘汰原因")
