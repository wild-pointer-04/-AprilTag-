# 相机内参和畸变矫正详解

## 目录
1. [内参矩阵 K 的含义](#内参矩阵-k-的含义)
2. [畸变系数 dist 的含义](#畸变系数-dist-的含义)
3. [default_intrinsics() 的含义](#default_intrinsics-的含义)
4. [真实 K, dist 的含义和作用](#真实-k-dist-的含义和作用)
5. [投影矩阵 P 的应用](#投影矩阵-p-的应用)
6. [如何使用 cv2.undistort 观察矫正效果](#如何使用-cv2undistort-观察矫正效果)

---

## 内参矩阵 K 的含义

### 定义

内参矩阵 K 是一个 3×3 矩阵：

```
K = [fx  0  cx]
    [ 0 fy  cy]
    [ 0  0   1]
```

### 参数说明

- **fx, fy**: 焦距（像素单位）
  - 表示相机在 X 和 Y 方向的放大倍数
  - 通常 fx ≈ fy（像素是方形的）
  - 单位：像素
  - 物理意义：焦距越长，视野越小，物体越大

- **cx, cy**: 主点坐标（像素）
  - 表示光轴与图像平面的交点
  - 理想情况下：cx = w/2, cy = h/2（图像中心）
  - 实际中可能略有偏移（镜头安装误差）

- **0, 1**: 固定值
  - 用于齐次坐标变换

### 作用

1. **3D 到 2D 投影**:
   ```
   [u]   [fx  0  cx] [X]
   [v] = [ 0 fy  cy] [Y]
   [w]   [ 0  0   1] [Z]
   
   像素坐标: (u/w, v/w)
   ```

2. **描述相机视野**:
   - 水平视野角: FOV_x = 2 × arctan(w / (2×fx))
   - 垂直视野角: FOV_y = 2 × arctan(h / (2×fy))

3. **位姿估计（PnP）**:
   - 将 3D 点投影到 2D 图像平面
   - 用于计算相机相对于物体的位姿

---

## 畸变系数 dist 的含义

### 定义

畸变系数通常包含 5 个参数（OpenCV 标准）：

```
dist = [k1, k2, p1, p2, k3]
```

### 参数说明

#### 径向畸变（Radial Distortion）

- **k1, k2, k3**: 径向畸变系数
  - 描述镜头中心到边缘的畸变程度
  - **k1 > 0**: 桶形畸变（barrel distortion）
    - 图像边缘向内弯曲
    - 常见于广角镜头
  - **k1 < 0**: 枕形畸变（pincushion distortion）
    - 图像边缘向外弯曲
    - 常见于长焦镜头

#### 切向畸变（Tangential Distortion）

- **p1, p2**: 切向畸变系数
  - 描述镜头与图像平面不平行造成的畸变
  - 通常很小，可以忽略

### 畸变模型

OpenCV 使用的畸变模型：

```
x_corrected = x × (1 + k1×r² + k2×r⁴ + k3×r⁶) + 2×p1×x×y + p2×(r² + 2×x²)
y_corrected = y × (1 + k1×r² + k2×r⁴ + k3×r⁶) + 2×p2×x×y + p1×(r² + 2×y²)

其中: r² = x² + y²
```

### 作用

- **矫正图像畸变**: 使弯曲的直线变直
- **提高测量精度**: 消除镜头物理缺陷的影响
- **改善视觉质量**: 使图像更符合人眼观察

---

## default_intrinsics() 的含义

### 定义

`default_intrinsics(h, w, f_scale=1.0)` 返回近似的内参：

```python
f = max(w, h) * f_scale
K = [[f, 0, w/2],
     [0, f, h/2],
     [0, 0, 1]]
dist = [0, 0, 0, 0, 0]  # 无畸变
```

### 假设

1. **焦距**: fx = fy = max(w, h) × f_scale
   - 近似为图像对角线长度
   - 这是一个粗略估计

2. **主点**: cx = w/2, cy = h/2
   - 假设主点在图像中心
   - 实际可能略有偏移

3. **畸变**: dist = [0, 0, 0, 0, 0]
   - 假设无畸变
   - 实际镜头都有一定畸变

### 用途

- **快速测试**: 当没有真实内参时的临时方案
- **粗略估计**: 仅用于判断是否歪斜，不用于精确测量
- **开发调试**: 在开发阶段快速验证算法

### 局限性

- ❌ 不能用于精确的位姿估计
- ❌ 不能用于畸变矫正
- ❌ 不适用于需要高精度的应用

---

## 真实 K, dist 的含义和作用

### 获取方式

1. **相机标定**:
   - 使用 OpenCV 的 `camera_calibration` 工具
   - 使用棋盘格或圆点板进行标定
   - 需要多张不同角度的标定图像（通常 15-20 张）

2. **从 ROS2 CameraInfo 提取**:
   - 如果相机已经标定过
   - 从 ROS2 话题中提取内参
   - 本项目提供了 `camera_rectifier.py` 工具

3. **厂商提供**:
   - 某些工业相机提供标定数据
   - 但通常需要自己标定以获得最佳精度

### 作用

#### 1. 精确的 3D-2D 投影

```python
# 将 3D 点投影到 2D 图像
points_3d = np.array([[X, Y, Z], ...])  # 3D 点
points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist)
```

#### 2. 精确的畸变矫正

```python
# 矫正图像畸变
undistorted = cv2.undistort(image, K, dist)
```

#### 3. 精确的位姿估计（PnP）

```python
# 从 2D-3D 对应点估计位姿
success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist)
```

#### 4. 提高测量精度

- 消除镜头畸变的影响
- 准确计算物体尺寸
- 精确估计相机位姿

---

## 投影矩阵 P 的应用

### 定义

投影矩阵 P 是一个 3×4 矩阵：

```
P = [fx  0  cx  tx]
    [ 0 fy  cy  ty]
    [ 0  0   1   0]
```

### 与内参矩阵 K 的关系

- 前 3×3 部分等于内参矩阵 K（如果 tx = ty = 0）
- 增加了平移项 tx, ty

### 应用场景

#### 1. 双目视觉（Stereo Vision）

```python
# 左右相机的投影矩阵
P_left = [fx_l  0  cx_l  tx_l]
         [ 0  fy_l  cy_l  ty_l]
         [ 0    0    1     0  ]

P_right = [fx_r  0  cx_r  tx_r]
          [ 0  fy_r  cy_r  ty_r]
          [ 0    0    1     0  ]

# 将 3D 点投影到左右图像
point_3d = [X, Y, Z, 1]
point_2d_left = P_left @ point_3d
point_2d_right = P_right @ point_3d
```

#### 2. 已矫正图像（Rectified Image）

- 经过立体矫正后的图像
- P 矩阵包含了矫正后的内参
- 可以直接用于立体匹配

#### 3. 3D 点投影

```python
# 将 3D 点投影到 2D 图像坐标
point_3d = np.array([X, Y, Z, 1])  # 齐次坐标
point_2d_homogeneous = P @ point_3d  # [u, v, w]
point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]  # (u/w, v/w)
```

### 在本项目中的应用

- 通常使用 **K 和 dist** 进行畸变矫正和 PnP
- **P 矩阵**主要用于双目视觉系统
- 如果使用已矫正的图像，可以用 P 的前 3×3 部分作为 K

---

## 如何使用 cv2.undistort 观察矫正效果

### 基本用法

```python
import cv2
import numpy as np

# 读取图像
img = cv2.imread("image.jpg")

# 加载内参和畸变系数
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float64)
dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

# 畸变矫正
undistorted = cv2.undistort(img, K, dist)

# 保存结果
cv2.imwrite("undistorted.jpg", undistorted)
```

### 使用本项目提供的工具

```bash
# 使用真实内参进行畸变矫正
python src/undistort_demo.py --image data/board.png

# 查看详细说明
python src/undistort_demo.py --explain
```

### 观察要点

1. **绘制网格线**:
   - 在图像上绘制规则的网格线
   - 观察矫正前后直线的变化

2. **对比原始图像和矫正图像**:
   - 原始图像：直线应该是弯曲的（如果有畸变）
   - 矫正图像：直线应该变直

3. **检查边缘区域**:
   - 畸变通常在图像边缘最明显
   - 观察边缘的直线是否变直

### 示例代码

```python
import cv2
import numpy as np

def draw_grid(img, spacing=50, color=(0, 255, 0), thickness=2):
    """在图像上绘制网格线"""
    h, w = img.shape[:2]
    vis = img.copy()
    
    # 垂直线
    for x in range(0, w, spacing):
        cv2.line(vis, (x, 0), (x, h), color, thickness)
    
    # 水平线
    for y in range(0, h, spacing):
        cv2.line(vis, (0, y), (w, y), color, thickness)
    
    return vis

# 读取图像
img = cv2.imread("image.jpg")

# 加载内参
K, dist = load_camera_intrinsics("config/camera_info.yaml")

# 绘制网格
img_with_grid = draw_grid(img)

# 畸变矫正
undistorted = cv2.undistort(img, K, dist)
undistorted_with_grid = draw_grid(undistorted)

# 对比
comparison = np.hstack([img_with_grid, undistorted_with_grid])
cv2.imwrite("comparison.png", comparison)
```

### 判断矫正效果

- ✅ **成功**: 矫正后直线变直，网格线规则
- ❌ **失败**: 矫正后直线仍然弯曲，或出现新的畸变
  - 可能原因：内参不准确、畸变模型不匹配

---

## 总结

| 项目 | default_intrinsics() | 真实 K, dist |
|------|---------------------|--------------|
| **精度** | 近似值 | 精确值 |
| **畸变** | 假设无畸变 | 考虑真实畸变 |
| **用途** | 快速测试、粗略估计 | 精确测量、位姿估计 |
| **获取** | 自动计算 | 需要标定或提取 |
| **适用场景** | 开发调试 | 生产环境 |

**建议**: 
- 开发阶段可以使用 `default_intrinsics()`
- 生产环境必须使用真实内参
- 定期重新标定以确保精度



