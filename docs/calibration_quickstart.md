# 相机标定和重投影误差分析 - 快速开始

## 功能

一键完成：
1. ✅ 使用 YAML 内参对 data 目录中的所有图像进行畸变矫正
2. ✅ 使用矫正后的图像进行相机标定
3. ✅ 计算所有图像的重投影误差
4. ✅ 绘制重投影误差分布散点图

## 快速使用

```bash
# 使用默认参数（推荐）
python src/calibration_and_reprojection.py

# 完整命令
python src/calibration_and_reprojection.py \
    --data-dir data \
    --undistorted-dir data_undistorted \
    --camera-yaml config/camera_info.yaml \
    --rows 15 \
    --cols 15 \
    --spacing 10.0 \
    --output outputs/reprojection_errors.png
```

## 输出结果

### 1. 矫正后的图像

保存在 `data_undistorted/` 目录：
```
data_undistorted/
├── board_undistorted.png
├── board1_undistorted.png
├── board2_undistorted.png
└── ...
```

### 2. 误差分布图

保存在 `outputs/reprojection_errors.png`：
- **上图**: 每张图像的平均误差散点图
- **下图**: 所有点的误差分布直方图

### 3. 控制台输出

```
✅ 总平均重投影误差: 0.0657 像素
   误差范围: 0.0499 ~ 0.0979 像素

统计信息:
  图像数量: 13
  总点数: 2925
  每张图像平均误差: 0.0657 ± 0.0130 像素
  所有点平均误差: 0.0657 ± 0.0988 像素
  最小误差: 0.0013 像素
  最大误差: 2.5508 像素
  中位数误差: 0.0547 像素
```

## 重投影误差说明

### 误差范围参考

- **< 0.5 像素**: ✅ 优秀，标定精度很高
- **0.5 - 1.0 像素**: ✅ 良好，标定精度较高
- **1.0 - 2.0 像素**: ⚠️ 一般，可以接受
- **> 2.0 像素**: ❌ 较差，可能需要重新标定

### 当前结果分析

从输出可以看到：
- 平均误差: **0.0657 像素** ✅ 非常优秀
- 误差范围: 0.0499 ~ 0.0979 像素 ✅ 非常稳定
- 所有点误差: 0.0657 ± 0.0988 像素 ✅ 分布良好

**结论**: 标定精度非常高，可以用于精确测量。

## 工作流程详解

### 步骤 1: 畸变矫正

1. 从 `config/camera_info.yaml` 加载内参
2. 如果图像尺寸不匹配，自动缩放内参
3. 使用 `cv2.undistort()` 矫正每张图像
4. 保存到 `data_undistorted/` 目录

### 步骤 2: 相机标定

1. 在矫正后的图像上检测 15×15 圆点网格
2. 使用检测到的 2D-3D 对应点进行标定
3. 调用 `cv2.calibrateCamera()` 计算新的内参

### 步骤 3: 重投影误差计算

对每张图像：
```python
# 将 3D 点投影到 2D 图像平面
imgpoints2, _ = cv2.projectPoints(
    objpoints[i], rvecs[i], tvecs[i], 
    camera_matrix, dist_coeffs
)

# 计算误差
error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
```

### 步骤 4: 误差可视化

生成两个图表：
- 散点图：每张图像的平均误差
- 直方图：所有点的误差分布

## 常见问题

### Q: 为什么有些图像检测失败？

**A**: 可能原因：
- 图像质量差（模糊、过曝）
- 圆点不完整（被遮挡）
- 网格尺寸不匹配

**解决**: 检查图像质量，确保圆点清晰可见。

### Q: 误差较大怎么办？

**A**: 可以尝试：
1. 增加标定图像数量（10-20 张）
2. 确保图像覆盖整个视野
3. 改善图像质量（光照、对焦）
4. 使用高质量的标定板

### Q: 矫正后的图像在哪里？

**A**: 保存在 `data_undistorted/` 目录，文件名格式：`原文件名_undistorted.png`

## 相关文档

- 详细使用说明: `docs/calibration_and_reprojection_usage.md`
- 内参说明: `docs/intrinsics_explanation.md`

