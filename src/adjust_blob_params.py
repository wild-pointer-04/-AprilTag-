"""
Blob 检测参数调整工具
根据观察结果（绿色点 vs 黄色点）调整检测参数
"""
import cv2


def build_blob_detector_tight():
    """
    严格的 blob 检测器（减少噪声）
    
    适用场景：
    - 绿色点太多（明显超过圆点数量）
    - 检测到很多背景噪声
    
    调整方向：提高阈值，过滤更多噪声
    """
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True
    p.minArea = 15  # 提高最小面积（过滤小噪声）
    p.maxArea = 50000  # 降低最大面积（过滤大物体）
    p.filterByCircularity = True
    p.minCircularity = 0.6  # 提高圆形度要求（更严格）
    p.maxCircularity = 1.0
    p.filterByInertia = True
    p.minInertiaRatio = 0.20  # 提高惯性比要求（过滤椭圆）
    p.filterByConvexity = False
    return cv2.SimpleBlobDetector_create(p)


def build_blob_detector_loose():
    """
    宽松的 blob 检测器（减少漏检）
    
    适用场景：
    - 绿色点太少（某些圆点没有绿色点）
    - 漏掉了真正的圆点
    
    调整方向：降低阈值，检测更多候选点
    """
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True
    p.minArea = 5  # 降低最小面积（检测更小的圆点）
    p.maxArea = 200000  # 提高最大面积（允许更大的圆点）
    p.filterByCircularity = True
    p.minCircularity = 0.3  # 降低圆形度要求（允许椭圆）
    p.maxCircularity = 1.0
    p.filterByInertia = True
    p.minInertiaRatio = 0.10  # 降低惯性比要求（允许更椭的椭圆）
    p.filterByConvexity = False
    return cv2.SimpleBlobDetector_create(p)


def build_blob_detector_custom(min_area=10, max_area=100000, 
                               min_circularity=0.5, min_inertia_ratio=0.15):
    """
    自定义 blob 检测器
    
    Args:
        min_area: 最小面积（像素）
        max_area: 最大面积（像素）
        min_circularity: 最小圆形度 (0-1)
        min_inertia_ratio: 最小惯性比 (0-1)
    
    Returns:
        SimpleBlobDetector 对象
    """
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True
    p.minArea = min_area
    p.maxArea = max_area
    p.filterByCircularity = True
    p.minCircularity = min_circularity
    p.maxCircularity = 1.0
    p.filterByInertia = True
    p.minInertiaRatio = min_inertia_ratio
    p.filterByConvexity = False
    return cv2.SimpleBlobDetector_create(p)


def print_parameter_guide():
    """
    打印参数调整指南
    """
    print("""
=== Blob 检测参数调整指南 ===

根据观察结果（绿色点 vs 黄色点）调整参数：

1. 如果绿色点太多（明显超过圆点数量）：
   → 使用 build_blob_detector_tight()
   → 或提高 minArea, minCircularity, minInertiaRatio

2. 如果绿色点太少（某些圆点没有绿色点）：
   → 使用 build_blob_detector_loose()
   → 或降低 minArea, minCircularity, minInertiaRatio

3. 如果绿色点和黄色点位置偏差大（>10像素）：
   → 提高 minCircularity（更严格的圆形度）
   → 或使用更好的精化方法

4. 如果黄色点数量不足（< rows×cols）：
   → 先确保所有圆点都有绿色点
   → 然后检查网格尺寸是否正确

参数说明：
- minArea: 最小面积（像素），过滤小噪声
- maxArea: 最大面积（像素），过滤大物体
- minCircularity: 最小圆形度 (0-1)，1.0=完美圆
- minInertiaRatio: 最小惯性比 (0-1)，过滤椭圆形状


""")


if __name__ == "__main__":
    print_parameter_guide()

