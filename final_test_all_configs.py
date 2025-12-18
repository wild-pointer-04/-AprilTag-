#!/usr/bin/env python3
"""
测试所有可能的配置组合
"""

import cv2
import sys
sys.path.append('.')

from src.apriltag_guided_grid_detector import AprilTagGuidedGridDetector

# 测试多种配置
configs = [
    {
        'name': 'A1: 圆点65mm, AprilTag71mm, 外部',
        'circle_spacing': 0.065,
        'apriltag_size': 0.0714,
        'apriltag_position': 'right_top'
    },
    {
        'name': 'A2: 圆点65mm, AprilTag71mm, 内部',
        'circle_spacing': 0.065,
        'apriltag_size': 0.0714,
        'apriltag_position': 'right_top_inside'
    },
    {
        'name': 'B1: 圆点6.5mm, AprilTag7.1mm, 外部',
        'circle_spacing': 0.0065,
        'apriltag_size': 0.0071,
        'apriltag_position': 'right_top'
    },
    {
        'name': 'B2: 圆点6.5mm, AprilTag7.1mm, 内部',
        'circle_spacing': 0.0065,
        'apriltag_size': 0.0071,
        'apriltag_position': 'right_top_inside'
    },
]

image = cv2.imread('data/1764744101_27_picture.png')

if image is None:
    print("无法读取图像")
    exit(1)

print("="*80)
print("测试所有配置")
print("="*80)

results = []

for config in configs:
    print(f"\n测试: {config['name']}")
    print(f"  圆点间距: {config['circle_spacing']*1000:.1f}mm")
    print(f"  AprilTag: {config['apriltag_size']*1000:.1f}mm")
    print(f"  位置: {config['apriltag_position']}")
    
    detector = AprilTagGuidedGridDetector(
        pattern_size=(15, 15),
        circle_spacing=config['circle_spacing'],
        apriltag_size=config['apriltag_size'],
        max_match_distance=25.0,
        image_margin=20.0,
        apriltag_position=config['apriltag_position']
    )
    
    result = detector.detect(image)
    
    if result['success']:
        match_rate = result['match_count'] / result['valid_count'] * 100
        print(f"  ✅ 成功: {result['match_count']}/{result['valid_count']} ({match_rate:.1f}%)")
        
        results.append({
            'config': config,
            'match_count': result['match_count'],
            'valid_count': result['valid_count'],
            'match_rate': match_rate,
            'result': result,
            'detector': detector
        })
    else:
        print(f"  ❌ 失败: {result['message']}")

# 显示最佳结果
print("\n" + "="*80)
print("结果排名")
print("="*80)

if len(results) > 0:
    results.sort(key=lambda x: x['match_rate'], reverse=True)
    
    for i, res in enumerate(results):
        print(f"\n{i+1}. {res['config']['name']}")
        print(f"   匹配率: {res['match_rate']:.1f}%")
        print(f"   匹配数: {res['match_count']}/{res['valid_count']}")
    
    # 可视化最佳结果
    best = results[0]
    print(f"\n最佳配置: {best['config']['name']}")
    print(f"匹配率: {best['match_rate']:.1f}%")
    
    vis = best['detector'].visualize(image, best['result'], show_details=True)
    
    # 添加配置信息
    cv2.putText(vis, f"Best: {best['config']['name']}", 
               (10, vis.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imwrite('best_result.png', vis)
    cv2.imshow('Best Configuration', vis)
    
    print("\n按任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("\n❌ 所有配置都失败了！")
    print("\n请检查:")
    print("  1. AprilTag 是否能检测到？")
    print("  2. 圆点是否能检测到？")
    print("  3. 实际尺寸是否正确？")
