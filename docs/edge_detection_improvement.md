# 边缘圆点漏检原因分析与改进方案

## 问题现象

从检测结果看，**边缘圆点（特别是底部、左侧、右侧）**没有被识别，而**中心区域的圆点**都能正确识别。

## 根本原因分析

### 1. **透视畸变导致圆形度降低**
- **中心圆点**：接近垂直视角，保持较高圆形度
- **边缘圆点**：由于透视投影，变成**椭圆**，`circularity` 下降
- **当前过滤**：`minCircularity = 0.5` 可能过滤掉边缘的椭圆圆点

### 2. **部分遮挡**
- 边缘圆点容易被**板子边框**或**图像边界**部分遮挡
- 遮挡后的圆点**面积减小**、**形状不完整**，不符合 blob 检测条件

### 3. **Blob检测参数过严**
- `minInertiaRatio = 0.15` 对椭圆形状过滤严格
- 透视畸变后的椭圆圆点惯性比可能低于阈值

### 4. **网格匹配算法限制**
- `findCirclesGrid` 需要**完整的网格结构**
- 边缘点缺失会导致网格匹配失败，即使检测到了单独的 blob

### 5. **光照不均**
- 边缘区域可能存在**阴影**或**反光**
- 对比度不足影响 blob 检测

## 改进方案

### 方案1：放宽边缘区域的检测参数（推荐）

**思路**：对图像边缘区域使用更宽松的参数

```python
def build_adaptive_blob_detector_edge_aware(gray):
    """边缘感知的自适应blob检测器"""
    # 基础参数
    p = build_adaptive_blob_detector(gray)
    
    # 放宽参数以检测边缘的椭圆圆点
    p.minCircularity = 0.3  # 从0.5降到0.3（允许椭圆）
    p.minInertiaRatio = 0.10  # 从0.15降到0.10（允许更椭）
    p.minArea = 5  # 降低最小面积（允许部分遮挡）
    
    return p
```

### 方案2：多尺度检测

**思路**：在不同尺度下检测，提高边缘圆点检出率

```python
def multi_scale_detect(gray, rows, cols):
    """多尺度检测"""
    scales = [0.9, 1.0, 1.1]
    best_result = None
    best_score = 0
    
    for scale in scales:
        if scale != 1.0:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale)
        else:
            scaled = gray
        
        ok, centers = try_find(scaled, rows, cols)
        if ok:
            score = evaluate_detection_quality(centers, rows, cols)
            if score > best_score:
                best_score = score
                best_result = centers / scale  # 缩放回原图
    
    return best_result is not None, best_result
```

### 方案3：改进的网格匹配（允许缺失点）

**思路**：即使边缘点缺失，也能匹配网格

```python
def robust_grid_matching(blobs, rows, cols, max_missing=10):
    """鲁棒的网格匹配，允许部分缺失点"""
    # 1. 先检测所有可能的圆点（不限制严格匹配）
    # 2. 使用 RANSAC 拟合网格模型
    # 3. 允许部分点缺失（特别是边缘点）
    # 4. 对缺失点进行插值或标记
    pass
```

### 方案4：区域自适应检测

**思路**：中心区域用严格参数，边缘区域用宽松参数

```python
def region_adaptive_detect(gray, rows, cols):
    """区域自适应检测"""
    h, w = gray.shape
    center_region = gray[h//4:h*3//4, w//4:w*3//4]
    edge_regions = [
        gray[:h//4, :],  # 顶部
        gray[h*3//4:, :],  # 底部
        gray[:, :w//4],  # 左侧
        gray[:, w*3//4:],  # 右侧
    ]
    
    # 中心用严格参数
    center_detector = build_strict_detector()
    # 边缘用宽松参数
    edge_detector = build_relaxed_detector()
    
    # 分别检测然后合并
```

## 推荐实施顺序

1. **立即实施**：放宽 blob 检测参数（方案1）
   - 最简单，效果明显
   - 修改 `minCircularity = 0.3`, `minInertiaRatio = 0.10`

2. **短期改进**：添加多尺度检测（方案2）
   - 提高整体检测率
   - 对边缘区域特别有效

3. **长期优化**：实现鲁棒网格匹配（方案3）
   - 允许部分缺失点
   - 需要更多开发工作

