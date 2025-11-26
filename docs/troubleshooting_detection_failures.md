# 网格检测失败问题排查指南

## 问题：Blob数量足够但网格匹配失败

### 原因分析

`cv2.findCirclesGrid` 需要满足严格的几何约束才能成功匹配：

1. **完整的网格结构**
   - 所有点必须形成规则的行列排列
   - 边缘圆点缺失会导致匹配失败
   - 即使只缺少几个点，整个网格也可能无法识别

2. **几何一致性**
   - 相邻点间距必须基本一致
   - 行列必须保持平行和垂直关系
   - 透视畸变会破坏这种一致性

3. **透视畸变限制**
   - 相机角度太倾斜会导致透视变形过大
   - 远端的圆点会变得更小、更椭圆
   - OpenCV 的网格匹配算法对此有一定容忍度，但超过阈值就会失败

4. **点的顺序**
   - 必须能够建立明确的行列关系
   - 如果点的排列混乱，无法确定哪些点属于同一行/列

### 为什么 Blob 检测成功但网格匹配失败？

- **Blob 检测**：只关心单个圆点的特征（面积、圆度等），不关心点之间的关系
- **网格匹配**：需要所有点满足网格的几何约束，是一个全局优化问题

即使检测到足够数量的圆点，如果它们不满足网格约束，匹配就会失败。

## 已实施的改进

### 1. 超时和早期终止机制

**问题**：程序在搜索时尝试大量组合，导致运行时间过长

**解决方案**：
```python
# 在 smart_auto_search 中添加：
- max_attempts=50  # 最大尝试次数
- timeout_seconds=10.0  # 超时时间（秒）
```

**效果**：
- 避免无限循环
- 最多尝试 50 次组合
- 超过 10 秒自动停止

### 2. 优化的 Blob 检测参数

**改进前**：
```python
minArea: 300
maxArea: 1000
minCircularity: 0.3
minInertiaRatio: 0.15
```

**改进后**：
```python
minArea: 200          # 降低，适应远距离小圆点
maxArea: 5000         # 提高，适应近距离大圆点
minCircularity: 0.2   # 降低，适应透视畸变
minInertiaRatio: 0.1  # 降低，适应椭圆形变
```

**新增功能**：
- 多阈值检测（10-220，步长10）
- 颜色过滤（检测暗色圆点）
- 更宽松的形状要求

### 3. 图像预处理

**功能**：
```python
def preprocess_for_detection(gray, enhance_contrast=True, denoise=True):
    # 1. 双边滤波去噪（保留边缘）
    # 2. CLAHE 对比度增强
```

**效果**：
- 改善低对比度图像
- 减少噪声干扰
- 保留圆点边缘

### 4. 跳帧和最大帧数限制

**新增参数**：
```bash
--skip-frames 5      # 每隔5帧处理一次
--max-frames 100     # 最多处理100帧
```

**效果**：
- 加快处理速度
- 减少重复计算
- 适合快速验证

## 使用建议

### 1. 快速测试（推荐）

处理少量帧，快速验证：
```bash
python src/tilt_checker_node.py \
  --rosbag rosbags/testbag \
  --image-topic /camera/color/image_raw \
  --camera-yaml config/camera_info.yaml \
  --save-images \
  --output-dir outputs/quick_test \
  --max-frames 10 \
  --skip-frames 5
```

### 2. 完整处理

处理所有帧（可能需要较长时间）：
```bash
python src/tilt_checker_node.py \
  --rosbag rosbags/testbag \
  --image-topic /camera/color/image_raw \
  --camera-yaml config/camera_info.yaml \
  --save-images \
  --output-dir outputs/full_analysis
```

### 3. 单张图像测试

测试特定图像的检测效果：
```bash
python test_detection_quick.py outputs/testbag_analysis/images/frame_000001_result.png
```

## 如何改善检测成功率

### 1. 调整相机位置

**最重要的因素**：
- ✅ 相机尽量正对标定板（减少透视畸变）
- ✅ 保持适当距离（圆点不要太小或太大）
- ✅ 确保标定板完整在视野内
- ✅ 避免边缘圆点被遮挡或裁剪

**推荐角度**：
- 倾斜角度 < 30°（最好 < 20°）
- 旋转角度 < 45°（最好 < 30°）

### 2. 改善光照条件

**关键要素**：
- ✅ 均匀照明（避免阴影和反光）
- ✅ 足够的对比度（圆点与背景）
- ✅ 避免过曝或欠曝
- ✅ 柔和的漫射光（避免强烈的直射光）

### 3. 检查标定板质量

**要求**：
- ✅ 圆点清晰、边缘锐利
- ✅ 圆点大小一致
- ✅ 间距均匀
- ✅ 无污损、褶皱、变形

### 4. 调整 Blob 检测参数

如果默认参数不适合你的场景，可以在 `src/utils.py` 中调整：

```python
BLOB_DETECTOR_PARAMS = {
    "minArea": 200,      # 根据圆点大小调整
    "maxArea": 5000,     # 根据圆点大小调整
    "minCircularity": 0.2,  # 降低以适应更大的畸变
    "minInertiaRatio": 0.1,  # 降低以适应更椭圆的形状
}
```

**调整建议**：
- 圆点太小检测不到 → 降低 `minArea`
- 检测到太多噪声 → 提高 `minArea` 和 `minCircularity`
- 透视畸变大 → 降低 `minCircularity` 和 `minInertiaRatio`

### 5. 使用图像预处理

如果图像质量不佳，可以启用预处理：

在 `src/detect_grid_improved.py` 的 `try_find_adaptive` 中：
```python
ok, centers, keypoints = try_find_adaptive(
    gray, rows, cols, 
    symmetric=True,
    use_preprocessing=True  # 启用预处理
)
```

## 常见错误和解决方案

### 错误 1：检测到的点数远少于期望

**现象**：
```
[DEBUG] Blob检测: 找到 50 个候选圆点, 期望 225 个点
```

**原因**：
- Blob 检测参数太严格
- 光照不足或对比度低
- 圆点太小或太模糊

**解决方案**：
1. 降低 `minArea`
2. 降低 `minCircularity`
3. 改善光照
4. 调整相机距离

### 错误 2：检测到的点数足够但匹配失败

**现象**：
```
[DEBUG] Blob检测: 找到 185 个候选圆点, 期望 225 个点
[DEBUG] 原因: Blob数量足够但网格匹配失败
```

**原因**：
- 网格结构不完整（边缘缺失）
- 透视畸变过大
- 检测到太多噪声点

**解决方案**：
1. 调整相机角度，减少透视畸变
2. 确保标定板完整在视野内
3. 提高 Blob 检测的选择性（提高 `minCircularity`）
4. 使用更小的网格尺寸（如 13x13 代替 15x15）

### 错误 3：程序运行时间过长

**现象**：
- 程序一直运行不停止
- 大量 "匹配失败" 消息

**原因**：
- 自动搜索尝试了太多组合
- 每次尝试都失败

**解决方案**：
1. 使用 `--max-frames` 限制处理帧数
2. 使用 `--skip-frames` 跳帧处理
3. 改善检测条件（见上文）
4. 程序已自动添加超时机制（10秒）

## 性能优化建议

### 1. 快速验证

```bash
# 只处理前10帧，每隔5帧处理一次
--max-frames 10 --skip-frames 5
```

### 2. 批量处理

```bash
# 每隔10帧处理一次，适合长时间录制
--skip-frames 10
```

### 3. 关闭图像保存

```bash
# 不保存图像，只保存结果数据
# （去掉 --save-images 参数）
```

## 总结

**关键要点**：

1. ✅ **相机位置最重要**：尽量正对标定板，减少透视畸变
2. ✅ **光照条件很关键**：均匀照明，足够对比度
3. ✅ **使用快速测试**：先用少量帧验证，再完整处理
4. ✅ **调整参数**：根据实际情况调整 Blob 检测参数
5. ✅ **程序已优化**：添加了超时、跳帧、最大帧数等功能

**如果仍然失败**：
- 检查标定板是否完整可见
- 尝试更小的网格尺寸（如 11x11）
- 使用单张图像测试工具 `test_detection_quick.py`
- 查看保存的图像，确认圆点是否清晰可见
