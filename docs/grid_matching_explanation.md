# 网格匹配详解：原理、代码位置与算法

## 什么是网格匹配？

**网格匹配**是将检测到的**无序圆点**组织成**有序网格结构**的过程。

### 问题描述

**输入**：
- 图像中检测到的圆点（无序，只知道位置）
- 期望的网格尺寸：`rows×cols`（如 15×15）

**输出**：
- 按行列顺序排列的有序点集
- 每个点对应网格中的特定位置（第几行第几列）

### 为什么需要网格匹配？

1. **PnP 求解需要**：需要知道每个 2D 点对应哪个 3D 点
2. **建立对应关系**：图像中的点 ↔ 板子坐标系中的点
3. **排序输出**：按照网格的行列顺序排列

## 代码位置

### 主要代码位置

网格匹配的核心代码在 **OpenCV 内部**（C++ 实现），Python 中通过以下函数调用：

```python
# 位置：src/detect_grid_improved.py 第 216 行
ok, centers = cv2.findCirclesGrid(gray, (cols, rows), flags=flags, blobDetector=det)
```

**调用链**：
```
run_tilt_check.py 
  → process_one()
    → try_find_func()  (即 try_find_adaptive)
      → cv2.findCirclesGrid()  ← 网格匹配在这里！
```

### 相关代码文件

1. **`src/detect_grid_improved.py`** (第 192-233 行)
   - `try_find_adaptive()`: 调用 `findCirclesGrid`
   - 包含诊断输出

2. **`src/detect_grid.py`** (第 6-11 行)
   - `try_find()`: 原版实现

3. **OpenCV 源码**（C++）
   - 位置：`opencv/modules/calib3d/src/calibinit.cpp`
   - 函数：`findCirclesGrid()` 或 `findCirclesGridDefault()`

## 网格匹配原理

### 整体流程

```
无序圆点集合 → 聚类分析 → 网格拟合 → 几何验证 → 排序输出
```

### 详细算法步骤

#### 步骤1：聚类分析（CALIB_CB_CLUSTERING）

**目的**：识别哪些圆点可能属于同一网格

**算法**：

```python
# 伪代码
def clustering_analysis(keypoints):
    """
    将圆点按空间位置聚类
    """
    # 1. 计算所有点之间的距离
    distances = compute_pairwise_distances(keypoints)
    
    # 2. 找到相邻点（距离最近的点）
    neighbors = find_nearest_neighbors(distances, threshold)
    
    # 3. 识别网格模式
    # - 同一行的点：水平距离相近
    # - 同一列的点：垂直距离相近
    # - 相邻点：距离最小
    
    # 4. 构建连接图
    graph = build_connectivity_graph(keypoints, neighbors)
    
    return clusters
```

**关键**：
- 不需要知道确切的网格尺寸
- 只识别"可能的网格模式"
- 基于空间距离和几何关系

#### 步骤2：网格拟合（Grid Fitting）

**目的**：将聚类后的点组织成 `rows×cols` 的网格

**算法**：

```python
# 伪代码
def grid_fitting(clustered_points, rows, cols):
    """
    将点组织成 rows×cols 的网格
    """
    # 1. 找到四个角点（或关键点）
    corners = find_corner_points(clustered_points)
    # 可能策略：
    # - 最左上、最右上、最左下、最右下
    # - 或者使用凸包算法
    
    # 2. 建立初始网格模型
    # 假设网格是规则的（透视投影后可能变形）
    grid_model = initialize_grid_model(corners, rows, cols)
    
    # 3. 迭代匹配
    for iteration in range(max_iterations):
        # 3.1 预测每个网格位置的点
        predicted_positions = grid_model.predict_all_positions()
        
        # 3.2 将实际点匹配到最近的预测位置
        matches = match_points_to_positions(clustered_points, predicted_positions)
        
        # 3.3 更新网格模型（使用匹配的点）
        grid_model.update(matches)
        
        # 3.4 检查收敛
        if converged(matches):
            break
    
    return grid_model, matches
```

**关键步骤**：

1. **初始匹配**：
   - 找到网格的边界点（四个角）
   - 建立初始的网格模型（可能是透视变换）

2. **迭代优化**：
   - 预测每个网格位置应该在哪里
   - 将实际检测到的点匹配到最近的预测位置
   - 更新网格模型（调整透视参数）
   - 重复直到收敛

3. **几何验证**：
   - 检查相邻点距离是否一致
   - 检查行列是否平行
   - 验证网格的几何一致性

#### 步骤3：几何验证

**目的**：确保匹配的网格符合几何约束

**验证项**：

```python
# 伪代码
def geometric_validation(matched_grid, rows, cols):
    """
    验证网格的几何一致性
    """
    # 1. 检查相邻点距离
    row_distances = compute_row_distances(matched_grid)
    col_distances = compute_col_distances(matched_grid)
    
    # 距离应该大致相等（允许一定误差）
    if std(row_distances) > threshold:
        return False  # 距离不一致
    
    # 2. 检查行列角度
    row_angles = compute_row_angles(matched_grid)
    col_angles = compute_col_angles(matched_grid)
    
    # 同一行的点应该大致水平对齐
    if std(row_angles) > threshold:
        return False  # 行不平行
    
    # 3. 检查完整性
    if len(matched_grid) < rows * cols * 0.9:  # 允许10%缺失
        return False  # 缺失点太多
    
    return True
```

#### 步骤4：排序输出

**目的**：按照网格的行列顺序排列点

**排序规则**：

```python
# 伪代码
def sort_points(matched_grid, rows, cols):
    """
    按行列顺序排序点
    """
    sorted_points = []
    
    # 对称网格：从左上角开始，按行扫描
    for row in range(rows):
        for col in range(cols):
            point = matched_grid[row, col]
            sorted_points.append(point)
    
    # 非对称网格：考虑错位
    # ...
    
    return sorted_points
```

**输出格式**：
- 形状：`(rows*cols, 1, 2)`
- 顺序：`(0,0), (0,1), ..., (0,cols-1), (1,0), (1,1), ...`

## OpenCV 内部实现原理

### 基于聚类的算法（CALIB_CB_CLUSTERING）

OpenCV 的 `findCirclesGrid` 使用以下策略：

1. **距离聚类**：
   - 计算所有点之间的欧氏距离
   - 找到距离最近的点对（相邻点）
   - 构建连接图

2. **网格识别**：
   - 识别"行"和"列"的模式
   - 基于点的空间分布和连接关系

3. **迭代匹配**：
   - 使用 RANSAC 或类似算法
   - 尝试不同的网格配置
   - 选择最佳匹配

### 算法复杂度

- **时间复杂度**：O(n²) 或更高（n 是圆点数量）
- **空间复杂度**：O(n²)（存储距离矩阵）

## 为什么网格匹配会失败？

### 常见失败原因

1. **Blob 数量不足**
   - 检测到的圆点 < `rows×cols`
   - 无法建立完整的网格结构

2. **结构不完整**
   - 边缘点缺失（特别是四个角点）
   - 无法确定网格边界

3. **透视畸变过大**
   - 板子倾斜角度 > 45°
   - 圆点间距变化太大
   - 行列不再平行

4. **几何不一致**
   - 相邻点距离差异太大
   - 行列角度偏差太大
   - 不符合网格的几何约束

5. **行列数不匹配**
   - 实际是 10×10，但传入 15×15
   - 算法无法匹配

## 改进思路

### 1. 鲁棒网格匹配（允许缺失点）

```python
def robust_grid_matching(keypoints, rows, cols, max_missing=10):
    """
    允许部分点缺失的网格匹配
    """
    # 1. 使用 RANSAC 拟合网格模型
    # 2. 允许部分点缺失（特别是边缘点）
    # 3. 对缺失点进行插值
    pass
```

### 2. 多尺度匹配

```python
def multi_scale_matching(image, rows, cols):
    """
    在不同尺度下尝试匹配
    """
    for scale in [0.8, 1.0, 1.2]:
        scaled = resize(image, scale)
        ok, centers = findCirclesGrid(scaled, rows, cols)
        if ok:
            return ok, centers / scale
    return False, None
```

### 3. 基于模板的匹配

```python
def template_based_matching(image, template_grid):
    """
    使用模板匹配网格
    """
    # 1. 定义模板网格（已知的网格结构）
    # 2. 在图像中搜索匹配的网格
    # 3. 使用模板匹配算法
    pass
```

## 总结

### 网格匹配的本质

**输入**：无序的点集合 + 期望的网格尺寸  
**输出**：有序的点集合（按行列排列）  
**核心**：建立"检测到的点" ↔ "网格位置"的对应关系

### 代码位置

- **调用位置**：`src/detect_grid_improved.py:216`
- **实现位置**：OpenCV 内部（C++）
- **相关代码**：`src/detect_grid.py`, `src/detect_grid_improved.py`

### 关键原理

1. **聚类分析**：识别网格模式
2. **网格拟合**：将点组织成网格
3. **几何验证**：确保几何一致性
4. **排序输出**：按行列顺序排列

### 失败原因

- Blob 数量不足
- 结构不完整
- 透视畸变过大
- 几何不一致
- 行列数不匹配

