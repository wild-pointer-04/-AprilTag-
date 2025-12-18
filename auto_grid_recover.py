#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动恢复网格行列编号（支持遮挡、不规则 blob、列缺失）

输入：
    filtered_blobs: Nx2 数组，绿色 blob 像素坐标
    origin: 网格起始点
    row_dir, col_dir: 单位向量
    spacing: 行列间距
    rows, cols: 网格大小

输出：
    ordered_grid: rows x cols x 2
"""

import numpy as np
import cv2


# ---------------------------------------
#  构建规则网格
# ---------------------------------------
def build_grid(origin, row_dir, col_dir, spacing, rows, cols):
    grid = np.zeros((rows, cols, 2), dtype=float)

    for r in range(rows):
        for c in range(cols):
            grid[r, c] = origin + r * spacing * row_dir + c * spacing * col_dir

    return grid


# ---------------------------------------
#  为每个 blob 找到最近的网格点
# ---------------------------------------
def assign_row_col(blob_points, grid_points, max_dist=50):
    rows, cols = grid_points.shape[:2]
    assignments = []

    for i, blob in enumerate(blob_points):
        min_dist = float("inf")
        best = (-1, -1)

        for r in range(rows):
            for c in range(cols):
                gp = grid_points[r, c]
                dist = np.linalg.norm(blob - gp)

                if dist < min_dist:
                    min_dist = dist
                    best = (r, c)

        if min_dist < max_dist:  # 限制最大距离
            assignments.append({
                "blob_idx": i,
                "row": best[0],
                "col": best[1],
                "dist": min_dist
            })

    return assignments


# ---------------------------------------
#  生成行列排序后的 blob 网格
# ---------------------------------------
def build_ordered_grid(assignments, blob_points, rows, cols):
    ordered = np.full((rows, cols, 2), np.nan)

    for a in assignments:
        r, c = a["row"], a["col"]
        ordered[r, c] = blob_points[a["blob_idx"]]

    return ordered


# ---------------------------------------
# 可选：可视化查看匹配结果
# ---------------------------------------
def visualize(image, ordered_grid):
    img = image.copy()

    rows, cols = ordered_grid.shape[:2]
    for r in range(rows):
        for c in range(cols):
            p = ordered_grid[r, c]
            if not np.isnan(p[0]):
                cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
                cv2.putText(img, f"{r},{c}", (int(p[0])+3, int(p[1])+3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    return img


# ---------------------------------------
# 主入口：自动恢复行列编号
# ---------------------------------------
def auto_recover_grid(filtered_blobs,
                      origin,
                      row_dir, col_dir,
                      spacing,
                      rows, cols,
                      visualize_image=None):

    # 1. 构建规则网格
    grid = build_grid(origin, row_dir, col_dir, spacing, rows, cols)

    # 2. 最近邻匹配 blob → grid
    assignments = assign_row_col(filtered_blobs, grid, max_dist=spacing * 1.5)

    # 3. 构建恢复后的 ordered blob grid
    ordered = build_ordered_grid(assignments, filtered_blobs, rows, cols)

    # 4. 可视化
    if visualize_image is not None:
        vis = visualize(visualize_image, ordered)
        return ordered, vis

    return ordered, None


# ------------------------------------------------
# 示例（填写你自己的 blob 输入即可）
# ------------------------------------------------
if __name__ == "__main__":
    # 假设你的 blob
    filtered_blobs = np.array([
        [100, 100],
        [140, 100],
        [180, 100],
        [100, 140],
        [140, 140],
        [180, 140],
    ])

    rows, cols = 3, 3
    spacing = 40
    origin = np.array([100, 100])

    row_dir = np.array([1, 0])  # 行方向
    col_dir = np.array([0, 1])  # 列方向

    ordered, _ = auto_recover_grid(filtered_blobs,
                                   origin,
                                   row_dir, col_dir,
                                   spacing,
                                   rows, cols)

    print("恢复后的 ordered grid：")
    print(ordered)
