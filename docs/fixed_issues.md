# 已修复的问题说明

## 问题 1: 固定检测为 15×15

### 问题描述
用户希望每次都使用 15×15 的圆点网格进行检测，而不是自动搜索。

### 解决方案
修改了 `src/run_tilt_check.py` 中的默认参数：

```python
parser.add_argument("--rows", type=int, default=15, help="圆点行数(内点)，默认15")
parser.add_argument("--cols", type=int, default=15, help="圆点列数(内点)，默认15")
```

**修改前**: `default=None`，如果不指定 `--rows` 和 `--cols`，会触发自动搜索。

**修改后**: `default=15`，默认使用 15×15 检测。

### 使用方式

```bash
# 使用默认的 15×15 检测（推荐）
python src/run_tilt_check.py --image data/board.png

# 如果需要自动搜索，使用 --auto 选项
python src/run_tilt_check.py --image data/board.png --auto

# 如果需要其他尺寸，显式指定
python src/run_tilt_check.py --image data/board.png --rows 12 --cols 13
```

## 问题 2: 图像尺寸不匹配（1280×720 vs 1920×1080）

### 问题描述
- YAML 文件中记录的分辨率是 **1280×720**（这是从 ROS2 CameraInfo 提取内参时的分辨率）
- 实际图像分辨率是 **1920×1080**
- 导致内参不匹配，影响位姿估计精度

### 原因分析
1. **1280×720 的来源**: 这是从 `config/camera_info.yaml` 文件中读取的，该文件是从 ROS2 CameraInfo 话题提取时保存的。当时相机发布的分辨率是 1280×720。
2. **为什么会出现不匹配**: 
   - 相机分辨率可能改变了
   - 或者提取内参时使用的是不同的分辨率设置
   - 或者图像经过了裁剪/缩放

### 解决方案
实现了**自动内参缩放功能**，当检测到图像尺寸不匹配时，自动按比例缩放内参矩阵。

#### 实现原理

内参矩阵中的参数需要按比例缩放：
- `fx_new = fx_old × (w_new / w_old)` - 焦距 X
- `fy_new = fy_old × (h_new / h_old)` - 焦距 Y  
- `cx_new = cx_old × (w_new / w_old)` - 主点 X
- `cy_new = cy_old × (h_new / h_old)` - 主点 Y

畸变系数通常不需要缩放（假设镜头不变）。

#### 代码实现

在 `src/utils.py` 中新增了 `scale_camera_intrinsics()` 函数：

```python
def scale_camera_intrinsics(K, dist, old_size, new_size):
    """根据图像尺寸变化缩放相机内参"""
    old_w, old_h = old_size
    new_w, new_h = new_size
    
    scale_x = new_w / old_w
    scale_y = new_h / old_h
    
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    
    return K_scaled, dist
```

在 `get_camera_intrinsics()` 函数中自动调用缩放功能（默认启用）。

#### 使用效果

**修改前**:
```
[WARN] 图像尺寸不匹配: YAML中为(1280, 720), 实际为(1920, 1080)
      继续使用 YAML 中的内参，但可能不够准确
```

**修改后**:
```
[INFO] 图像尺寸不匹配: YAML中为(1280, 720), 实际为(1920, 1080)
      自动缩放内参矩阵以适应新分辨率...
      缩放后的内参矩阵 K:
[[1.03572583e+03 0.00000000e+00 9.68266479e+02]
 [0.00000000e+00 1.03519244e+03 5.38415955e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
```

#### 缩放计算示例

原始内参（1280×720）:
- fx = 690.48
- fy = 690.13
- cx = 645.51
- cy = 358.94

缩放后内参（1920×1080）:
- fx = 690.48 × (1920/1280) = 1035.72
- fy = 690.13 × (1080/720) = 1035.19
- cx = 645.51 × (1920/1280) = 968.27
- cy = 358.94 × (1080/720) = 538.41

### 其他解决方案

如果不想使用自动缩放，可以：

1. **重新提取对应分辨率的内参**:
   ```bash
   # 确保相机以 1920×1080 分辨率发布 CameraInfo
   python src/camera_rectifier.py \
       --camera_info_topic /camera/color/camera_info \
       --output config/camera_info.yaml
   ```

2. **手动更新 YAML 文件中的图像尺寸**:
   ```yaml
   image_width: 1920
   image_height: 1080
   ```
   但这样内参矩阵不会自动缩放，仍然不够准确。

3. **为不同分辨率创建不同的 YAML 文件**:
   ```
   config/
   ├── camera_info_1280x720.yaml
   └── camera_info_1920x1080.yaml
   ```
   使用时指定对应的文件：
   ```bash
   python src/run_tilt_check.py --image data/board.png --camera-yaml config/camera_info_1920x1080.yaml
   ```

## 验证

运行以下命令验证修复：

```bash
# 验证 15×15 检测
python src/run_tilt_check.py --image data/board.png

# 应该看到：
# [DEBUG] findCirclesGrid: rows=15, cols=15, 对称网格
# 中点中心: (u,v)=(..., ...)  (rows=15, cols=15)

# 验证内参自动缩放
# 应该看到：
# [INFO] 图像尺寸不匹配: YAML中为(1280, 720), 实际为(1920, 1080)
#       自动缩放内参矩阵以适应新分辨率...
```

## 注意事项

1. **自动缩放假设**: 自动缩放假设相机镜头不变，只是分辨率改变。如果更换了镜头，应该重新标定。

2. **精度**: 自动缩放的内参精度取决于原始内参的精度。如果原始内参是在 1280×720 下精确标定的，缩放后的内参在 1920×1080 下应该也是准确的。

3. **畸变系数**: 畸变系数不进行缩放，因为它们是归一化的参数，不依赖于图像分辨率。

