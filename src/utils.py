import cv2
import numpy as np


# 统一的 blob 检测参数集中配置，所有检测流程共享
# 这些参数已经过优化，适应不同光照和透视条件
BLOB_DETECTOR_PARAMS = {
    "minArea": 20,           # 降低最小面积，适应透视畸变下的小圆点
    "maxArea": 150,        # 增大最大面积，适应透视畸变下的大圆点（之前60太小了！）
    "minCircularity": 0.2,  # 进一步降低圆度要求，适应椭圆变形
    "maxCircularity": 1.0,
    "minInertiaRatio": 0.2, # 进一步降低惯性比，适应严重变形
    "filterByConvexity": False,
    "minThreshold": 1,       # 降低起始阈值，提高检测敏感度
    "maxThreshold": 250,     # 提高最大阈值，适应不同光照
    "thresholdStep": 5,      # 减小步长，提高检测精度
}

def build_blob_detector():
    """
    构造圆点检测器，适用于常见对称圆点板。
    
    参数已优化以适应：
    - 不同光照条件（通过多阈值检测）
    - 透视畸变（降低圆度和惯性比要求）
    - 不同距离（宽松的面积范围）
    """
    p = cv2.SimpleBlobDetector_Params()
    
    # 颜色过滤（检测暗色圆点）
    p.filterByColor = True
    p.blobColor = 0  # 0=暗色, 255=亮色
    
    # 面积过滤
    p.filterByArea = True
    p.minArea = BLOB_DETECTOR_PARAMS["minArea"]
    p.maxArea = BLOB_DETECTOR_PARAMS["maxArea"]
    
    # 圆度过滤（降低要求以适应透视畸变）
    p.filterByCircularity = True
    p.minCircularity = BLOB_DETECTOR_PARAMS["minCircularity"]
    p.maxCircularity = BLOB_DETECTOR_PARAMS["maxCircularity"]
    
    # 惯性比过滤（降低要求以适应椭圆形变）
    p.filterByInertia = True
    p.minInertiaRatio = BLOB_DETECTOR_PARAMS["minInertiaRatio"]
    
    # 凸性过滤（关闭，因为透视畸变可能导致非凸）
    p.filterByConvexity = BLOB_DETECTOR_PARAMS["filterByConvexity"]
    
    # 多阈值检测（提高鲁棒性）
    p.minThreshold = BLOB_DETECTOR_PARAMS["minThreshold"]
    p.maxThreshold = BLOB_DETECTOR_PARAMS["maxThreshold"]
    p.thresholdStep = BLOB_DETECTOR_PARAMS["thresholdStep"]
    
    # 距离过滤（关闭，让 findCirclesGrid 处理）
    p.minDistBetweenBlobs = 1.0
    
    return cv2.SimpleBlobDetector_create(p)

def draw_axes(img, K, dist, rvec, tvec, axis_len=150):
    """在图像左下角附近绘制世界坐标轴（红X、绿Y、蓝Z）。"""
    axis = np.float32([
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, -axis_len],
        [0, 0, 0]
    ]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    imgpts = np.round(imgpts.reshape(-1, 2)).astype(int)
    o = tuple(int(v) for v in imgpts[3])
    cv2.line(img, o, tuple(int(v) for v in imgpts[0]), (0, 0, 255), 3)
    cv2.line(img, o, tuple(int(v) for v in imgpts[1]), (0, 255, 0), 3)
    cv2.line(img, o, tuple(int(v) for v in imgpts[2]), (255, 0, 0), 3)
    return img

def rvec_to_euler_xyz(rvec):
    angle_rad = np.linalg.norm(rvec)
    print("rvec:", rvec.ravel())
    print("rvec norm (rad):", angle_rad)
    print("rvec angle (deg):", np.degrees(angle_rad))
    """旋转向量 -> 欧拉角XYZ(度)。roll=X, pitch=Y, yaw=Z。"""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0.0
    return np.degrees([x, y, z])

def rvec_to_camera_tilt(rvec):
    """
    计算相机相对于水平面的倾斜角（假设板子水平放置）
    
    假设：
    - 板子是水平放置的（板子坐标系Z轴垂直向上）
    - 板子坐标系定义：X右、Y下、Z上（板子平面）
    - 相机坐标系：X右、Y下、Z前（光轴方向）
    
    返回：
    - roll: 相机左右倾斜（绕光轴Z旋转，度）
    - pitch: 相机前后倾斜（绕X轴旋转，度）
    - yaw: 相机水平旋转（绕Y轴旋转，度，通常不重要）
    
    注意：这是"相机相对于水平面"的角度，不是"板子相对于相机"的角度
    """
    R, _ = cv2.Rodrigues(rvec)
    
    # 板子坐标系 → 相机坐标系
    # 板子Z轴（垂直向上）在相机坐标系中的方向
    z_board_in_camera = R @ np.array([0.0, 0.0, 1.0])
    
    # 归一化
    z_board_in_camera = z_board_in_camera / np.linalg.norm(z_board_in_camera)
    
    # 计算倾斜角
    # Pitch: 平面旋转 = 板子Z轴在相机XZ平面的投影角度
    pitch = np.arctan2(-z_board_in_camera[2], z_board_in_camera[1]) * 180.0 / np.pi
    
    # Roll: 前后倾斜 = 板子Z轴在相机YZ平面的投影角度
    roll = np.arctan2(z_board_in_camera[0], z_board_in_camera[2]) * 180.0 / np.pi
    
    # Yaw: 左右倾斜（使用标准欧拉角）
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        yaw = np.arctan2(R[1,0], R[0,0]) * 180.0 / np.pi
    else:
        yaw = 0.0
    
    return roll, pitch, yaw

def normalize_angles(roll, pitch, yaw):
    """
    归一化角度到 [-180, 180] 范围，并处理Yaw接近180度的特殊情况
    """
    # 归一化到 [-180, 180]
    roll = ((roll + 180) % 360) - 180
    pitch = ((pitch + 180) % 360) - 180
    yaw = ((yaw + 180) % 360) - 180
    
    # 如果Yaw接近±180度，可能是板子翻面了
    # 这种情况下，Roll和Pitch的符号可能需要调整
    if abs(yaw) > 170:
        # 板子可能翻面了，调整角度
        roll = -roll
        pitch = -pitch
        yaw = yaw - 180 if yaw > 0 else yaw + 180
    
    return roll, pitch, yaw


def rvec_to_euler_zyx(rvec):
    """
    将旋转向量转换为ZYX欧拉角（内旋顺序：先绕Z轴旋转γ，再绕Y轴旋转α，再绕X轴旋转β）
    
    参数:
        rvec: 旋转向量 (3x1 numpy array)
    
    返回:
        (gamma, alpha, beta): ZYX欧拉角（弧度）
        - gamma: 绕Z轴旋转角度（弧度）
        - alpha: 绕Y轴旋转角度（弧度）
        - beta: 绕X轴旋转角度（弧度）
    
    注意：
        - 这是内旋顺序（intrinsic rotation）：R = Rz(γ) * Ry(α) * Rx(β)
        - 等价于外旋顺序（extrinsic rotation）：先绕X轴，再绕Y轴，最后绕Z轴
    """
    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    
    # 提取ZYX欧拉角（内旋顺序）
    # 从旋转矩阵中提取：
    # R = Rz(γ) * Ry(α) * Rx(β)
    # 
    # R = [cos(γ)cos(α)  -sin(γ)cos(β)+cos(γ)sin(α)sin(β)   sin(γ)sin(β)+cos(γ)sin(α)cos(β) ]
    #     [sin(γ)cos(α)   cos(γ)cos(β)+sin(γ)sin(α)sin(β)   -cos(γ)sin(β)+sin(γ)sin(α)cos(β) ]
    #     [-sin(α)        cos(α)sin(β)                       cos(α)cos(β)                    ]
    
    # 提取角度
    # alpha (绕Y轴旋转)
    alpha = np.arcsin(-R[2, 0])
    alpha = np.clip(alpha, -np.pi/2, np.pi/2)  # 限制在[-90°, 90°]范围内
    
    # 检查万向锁情况（cos(alpha) ≈ 0）
    if abs(np.cos(alpha)) > 1e-6:
        # 正常情况
        gamma = np.arctan2(R[1, 0], R[0, 0])  # 绕Z轴旋转
        beta = np.arctan2(R[2, 1], R[2, 2])   # 绕X轴旋转
    else:
        # 万向锁情况（alpha ≈ ±90°）
        # 此时gamma和beta不能唯一确定，我们选择beta=0
        gamma = np.arctan2(-R[0, 1], R[1, 1])
        beta = 0.0
    
    return gamma, alpha, beta


def compute_camera_to_board_transform(rvec, tvec):
    """
    计算从相机坐标系到标定板坐标系的变换参数
    
    参数:
        rvec: 旋转向量，表示标定板相对于相机的旋转（从标定板到相机的变换）
        tvec: 平移向量，表示标定板相对于相机的平移（从标定板到相机的变换）
    
    返回:
        (delta_x, delta_y, delta_z, gamma, alpha, beta): 变换参数
        - delta_x, delta_y, delta_z: 平移量（米）
        - gamma, alpha, beta: ZYX欧拉角（弧度）
    
    说明:
        OpenCV的solvePnP返回的是从标定板坐标系到相机坐标系的变换（R_board_to_cam, t_board_to_cam）
        我们需要计算从相机坐标系到标定板坐标系的变换（R_cam_to_board, t_cam_to_board）
        
        逆变换：
        - R_cam_to_board = R_board_to_cam^T
        - t_cam_to_board = -R_board_to_cam^T * t_board_to_cam
    """
    # 将旋转向量转换为旋转矩阵
    R_board_to_cam, _ = cv2.Rodrigues(rvec)
    
    # 计算逆变换（从相机到标定板）
    R_cam_to_board = R_board_to_cam.T  # 旋转矩阵的转置等于逆矩阵
    
    # 计算平移向量（从相机到标定板）
    t_cam_to_board = -R_cam_to_board @ tvec
    
    # 将旋转矩阵转换为旋转向量
    rvec_cam_to_board, _ = cv2.Rodrigues(R_cam_to_board)
    
    # 转换为ZYX欧拉角
    gamma, alpha, beta = rvec_to_euler_zyx(rvec_cam_to_board)
    
    # 提取平移量（单位：米，tvec已经是米单位了）
    delta_x = float(t_cam_to_board[0])
    delta_y = float(t_cam_to_board[1])
    delta_z = float(t_cam_to_board[2])
    
    return delta_x, delta_y, delta_z, gamma, alpha, beta

def load_camera_intrinsics(yaml_path: str = 'config/camera_info.yaml'):
    """
    从 YAML 文件加载相机内参和畸变系数。
    
    参数:
        yaml_path: YAML 文件路径（相对于项目根目录或绝对路径）
    
    返回:
        K: 内参矩阵 (3x3 numpy array)
        dist: 畸变系数 (1D numpy array)
        image_size: (width, height) 元组，如果 YAML 中有记录
    
    如果文件不存在或读取失败，返回 None, None, None
    """
    import yaml
    import os
    
    # 如果路径不是绝对路径，尝试从项目根目录查找
    if not os.path.isabs(yaml_path):
        # 尝试多个可能的路径
        possible_paths = [
            yaml_path,  # 当前目录
            os.path.join(os.path.dirname(os.path.dirname(__file__)), yaml_path),  # 项目根目录
        ]
        for path in possible_paths:
            if os.path.exists(path):
                yaml_path = path
                break
        else:
            print(f"[WARN] 未找到相机内参文件: {yaml_path}")
            return None, None, None
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # 提取内参矩阵
        if 'camera_matrix' not in data:
            print(f"[ERROR] YAML 文件中缺少 'camera_matrix' 字段")
            return None, None, None
        
        cam_matrix = data['camera_matrix']
        if 'data' in cam_matrix:
            K_data = cam_matrix['data']
            if len(K_data) == 9:
                K = np.array(K_data, dtype=np.float64).reshape(3, 3)
            else:
                print(f"[ERROR] 内参矩阵数据长度错误: 期望9，实际{len(K_data)}")
                return None, None, None
        else:
            print(f"[ERROR] YAML 文件中 'camera_matrix' 缺少 'data' 字段")
            return None, None, None
        
        # 提取畸变系数
        if 'distortion_coefficients' not in data:
            print(f"[WARN] YAML 文件中缺少 'distortion_coefficients' 字段，使用零畸变")
            dist = np.zeros(5, dtype=np.float64)
        else:
            dist_coeffs = data['distortion_coefficients']
            if 'data' in dist_coeffs:
                dist = np.array(dist_coeffs['data'], dtype=np.float64)
                # 确保至少有 5 个系数（OpenCV 标准）
                if len(dist) < 5:
                    dist = np.pad(dist, (0, 5 - len(dist)), mode='constant')
                elif len(dist) > 5:
                    dist = dist[:5]  # 只取前5个
            else:
                print(f"[WARN] YAML 文件中 'distortion_coefficients' 缺少 'data' 字段，使用零畸变")
                dist = np.zeros(5, dtype=np.float64)
        
        # 提取图像尺寸（可选）
        image_size = None
        if 'image_width' in data and 'image_height' in data:
            image_size = (int(data['image_width']), int(data['image_height']))
        
        print(f"[INFO] 成功加载相机内参: {yaml_path}")
        print(f"  内参矩阵 K:\n{K}")
        print(f"  畸变系数 D: {dist}")
        if image_size:
            print(f"  图像尺寸: {image_size[0]} x {image_size[1]}")
        
        return K, dist, image_size
        
    except FileNotFoundError:
        print(f"[WARN] 相机内参文件不存在: {yaml_path}")
        return None, None, None
    except yaml.YAMLError as e:
        print(f"[ERROR] 解析 YAML 文件失败: {e}")
        return None, None, None
    except Exception as e:
        print(f"[ERROR] 加载相机内参失败: {e}")
        return None, None, None


def default_intrinsics(h, w, f_scale=1.0):
    """
    没有相机内参时的近似（只用于判断是否歪斜）：
    f ≈ max(w,h)*f_scale, cx=w/2, cy=h/2, dist=0
    
    注意：如果存在 camera_info.yaml 文件，应该优先使用 load_camera_intrinsics() 加载真实内参。
    """
    f = max(w, h) * f_scale
    K = np.array([[f, 0, w/2.0],
                  [0, f, h/2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    dist = np.zeros(5)  # 假设无畸变
    return K, dist


def scale_camera_intrinsics(K, dist, old_size, new_size):
    """
    根据图像尺寸变化缩放相机内参。
    
    参数:
        K: 原始内参矩阵 (3x3)
        dist: 畸变系数（通常不需要缩放）
        old_size: (width, height) 原始图像尺寸
        new_size: (width, height) 新图像尺寸
    
    返回:
        K_scaled: 缩放后的内参矩阵
        dist: 畸变系数（不变）
    
    说明:
        内参矩阵中的 fx, fy, cx, cy 需要按比例缩放：
        - fx_new = fx_old * (w_new / w_old)
        - fy_new = fy_old * (h_new / h_old)
        - cx_new = cx_old * (w_new / w_old)
        - cy_new = cy_old * (h_new / h_old)
    """
    old_w, old_h = old_size
    new_w, new_h = new_size
    
    # 计算缩放比例
    scale_x = new_w / old_w
    scale_y = new_h / old_h
    
    # 缩放内参矩阵
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    
    return K_scaled, dist


def get_camera_intrinsics(h, w, yaml_path: str = 'config/camera_info.yaml', f_scale=1.0, auto_scale=True):
    """
    获取相机内参：优先从 YAML 文件加载，如果失败则使用默认值。
    
    参数:
        h: 图像高度（像素）
        w: 图像宽度（像素）
        yaml_path: YAML 文件路径
        f_scale: 如果使用默认内参，焦距缩放因子
        auto_scale: 如果图像尺寸不匹配，是否自动缩放内参（默认True）
    
    返回:
        K: 内参矩阵 (3x3 numpy array)
        dist: 畸变系数 (1D numpy array)
    """
    K, dist, image_size = load_camera_intrinsics(yaml_path)
    
    if K is not None and dist is not None:
        # 验证图像尺寸是否匹配（如果 YAML 中有记录）
        if image_size is not None:
            if image_size[0] != w or image_size[1] != h:
                if auto_scale:
                    # 自动缩放内参
                    print(f"[INFO] 图像尺寸不匹配: YAML中为{image_size}, 实际为({w}, {h})")
                    print(f"      自动缩放内参矩阵以适应新分辨率...")
                    K, dist = scale_camera_intrinsics(K, dist, image_size, (w, h))
                    print(f"      缩放后的内参矩阵 K:\n{K}")
                else:
                    print(f"[WARN] 图像尺寸不匹配: YAML中为{image_size}, 实际为({w}, {h})")
                    print(f"      继续使用 YAML 中的内参，但可能不够准确")
                    print(f"      提示: 可以重新提取对应分辨率的内参，或启用自动缩放功能")
        return K, dist
    else:
        # 使用默认内参
        print(f"[INFO] 使用默认内参（近似值）")
        return default_intrinsics(h, w, f_scale)
