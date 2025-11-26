#!/usr/bin/env python3
"""
PnP多解歧义解决器

专门解决AprilTag + 对称网格系统中的PnP多解问题
这是导致重投影误差高达247像素的根本原因

核心问题：
1. 对称网格模式导致多个可能的位姿解
2. PnP算法可能选择错误的解，导致巨大的重投影误差
3. AprilTag提供的约束信息没有被充分利用

解决策略：
1. 使用AprilTag位姿作为强约束
2. 多种PnP方法交叉验证
3. 几何一致性检查
4. 重投影误差阈值过滤
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PnPAmbiguityResolver:
    """PnP多解歧义解决器"""
    
    def __init__(self, 
                 max_reprojection_error: float = 10.0,
                 angle_consistency_threshold: float = 30.0):
        """
        初始化解决器
        
        Args:
            max_reprojection_error: 最大允许重投影误差(像素)
            angle_consistency_threshold: 角度一致性阈值(度)
        """
        self.max_reprojection_error = max_reprojection_error
        self.angle_consistency_threshold = angle_consistency_threshold
        
    def solve_robust_pnp_with_apriltag_constraint(self,
                                                  objpoints: np.ndarray,
                                                  imgpoints: np.ndarray,
                                                  camera_matrix: np.ndarray,
                                                  dist_coeffs: np.ndarray,
                                                  apriltag_rvec: np.ndarray,
                                                  apriltag_tvec: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, Dict]:
        """
        使用AprilTag约束的鲁棒PnP求解
        
        Args:
            objpoints: 3D物体点
            imgpoints: 2D图像点
            camera_matrix: 相机内参
            dist_coeffs: 畸变系数
            apriltag_rvec: AprilTag旋转向量
            apriltag_tvec: AprilTag平移向量
            
        Returns:
            (最佳rvec, 最佳tvec, 最佳误差, 详细信息)
        """
        solutions = []
        
        # 方法1: 直接使用AprilTag位姿作为初始猜测
        solution1 = self._solve_with_apriltag_guess(
            objpoints, imgpoints, camera_matrix, dist_coeffs,
            apriltag_rvec, apriltag_tvec
        )
        if solution1:
            solutions.append(solution1)
        
        # 方法2: 标准ITERATIVE方法
        solution2 = self._solve_standard_iterative(
            objpoints, imgpoints, camera_matrix, dist_coeffs
        )
        if solution2:
            solutions.append(solution2)
        
        # 方法3: P3P方法（如果点数足够）
        if len(objpoints) >= 4:
            solution3 = self._solve_p3p(
                objpoints, imgpoints, camera_matrix, dist_coeffs
            )
            if solution3:
                solutions.append(solution3)
        
        # 方法4: EPNP方法
        solution4 = self._solve_epnp(
            objpoints, imgpoints, camera_matrix, dist_coeffs
        )
        if solution4:
            solutions.append(solution4)
        
        # 方法5: 使用修正的AprilTag位姿作为约束
        solution5 = self._solve_with_corrected_apriltag_constraint(
            objpoints, imgpoints, camera_matrix, dist_coeffs,
            apriltag_rvec, apriltag_tvec
        )
        if solution5:
            solutions.append(solution5)
        
        # 评估所有解并选择最佳的
        best_solution = self._select_best_solution(
            solutions, objpoints, imgpoints, camera_matrix, dist_coeffs,
            apriltag_rvec, apriltag_tvec
        )
        
        if best_solution:
            method, rvec, tvec, error = best_solution
            info = {
                'method': method,
                'total_solutions': len(solutions),
                'all_errors': [s[3] for s in solutions],
                'apriltag_consistency': self._check_apriltag_consistency(
                    rvec, tvec, apriltag_rvec, apriltag_tvec
                )
            }
            return rvec, tvec, error, info
        else:
            return None, None, float('inf'), {'error': 'No valid solution found'}
    
    def _solve_with_apriltag_guess(self,
                                   objpoints: np.ndarray,
                                   imgpoints: np.ndarray,
                                   camera_matrix: np.ndarray,
                                   dist_coeffs: np.ndarray,
                                   apriltag_rvec: np.ndarray,
                                   apriltag_tvec: np.ndarray) -> Optional[Tuple]:
        """使用AprilTag位姿作为初始猜测"""
        try:
            success, rvec, tvec = cv2.solvePnP(
                objpoints, imgpoints, camera_matrix, dist_coeffs,
                rvec=apriltag_rvec.copy(),
                tvec=apriltag_tvec.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                error = self._calculate_reprojection_error(
                    rvec, tvec, objpoints, imgpoints, camera_matrix, dist_coeffs
                )
                return ('APRILTAG_GUESS', rvec, tvec, error)
        except Exception as e:
            logger.debug(f"AprilTag猜测方法失败: {e}")
        
        return None
    
    def _solve_standard_iterative(self,
                                  objpoints: np.ndarray,
                                  imgpoints: np.ndarray,
                                  camera_matrix: np.ndarray,
                                  dist_coeffs: np.ndarray) -> Optional[Tuple]:
        """标准迭代方法"""
        try:
            success, rvec, tvec = cv2.solvePnP(
                objpoints, imgpoints, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                error = self._calculate_reprojection_error(
                    rvec, tvec, objpoints, imgpoints, camera_matrix, dist_coeffs
                )
                return ('ITERATIVE', rvec, tvec, error)
        except Exception as e:
            logger.debug(f"标准迭代方法失败: {e}")
        
        return None
    
    def _solve_p3p(self,
                   objpoints: np.ndarray,
                   imgpoints: np.ndarray,
                   camera_matrix: np.ndarray,
                   dist_coeffs: np.ndarray) -> Optional[Tuple]:
        """P3P方法"""
        try:
            success, rvecs, tvecs = cv2.solvePnP(
                objpoints, imgpoints, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_P3P
            )
            
            if success:
                # P3P可能返回多个解，选择最佳的
                if isinstance(rvecs, list):
                    best_error = float('inf')
                    best_solution = None
                    
                    for rvec, tvec in zip(rvecs, tvecs):
                        error = self._calculate_reprojection_error(
                            rvec, tvec, objpoints, imgpoints, camera_matrix, dist_coeffs
                        )
                        if error < best_error:
                            best_error = error
                            best_solution = (rvec, tvec, error)
                    
                    if best_solution:
                        return ('P3P', best_solution[0], best_solution[1], best_solution[2])
                else:
                    error = self._calculate_reprojection_error(
                        rvecs, tvecs, objpoints, imgpoints, camera_matrix, dist_coeffs
                    )
                    return ('P3P', rvecs, tvecs, error)
        except Exception as e:
            logger.debug(f"P3P方法失败: {e}")
        
        return None
    
    def _solve_epnp(self,
                    objpoints: np.ndarray,
                    imgpoints: np.ndarray,
                    camera_matrix: np.ndarray,
                    dist_coeffs: np.ndarray) -> Optional[Tuple]:
        """EPNP方法"""
        try:
            success, rvec, tvec = cv2.solvePnP(
                objpoints, imgpoints, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if success:
                error = self._calculate_reprojection_error(
                    rvec, tvec, objpoints, imgpoints, camera_matrix, dist_coeffs
                )
                return ('EPNP', rvec, tvec, error)
        except Exception as e:
            logger.debug(f"EPNP方法失败: {e}")
        
        return None
    
    def _solve_with_corrected_apriltag_constraint(self,
                                                  objpoints: np.ndarray,
                                                  imgpoints: np.ndarray,
                                                  camera_matrix: np.ndarray,
                                                  dist_coeffs: np.ndarray,
                                                  apriltag_rvec: np.ndarray,
                                                  apriltag_tvec: np.ndarray) -> Optional[Tuple]:
        """使用修正的AprilTag约束"""
        try:
            # 对AprilTag位姿进行小幅调整，避免局部最优解
            for angle_offset in [0, 5, -5, 10, -10]:  # 度
                for trans_scale in [0.9, 1.0, 1.1]:
                    # 旋转调整
                    angle_rad = np.radians(angle_offset)
                    R_offset = cv2.Rodrigues(np.array([0, 0, angle_rad]))[0]
                    R_apriltag = cv2.Rodrigues(apriltag_rvec)[0]
                    R_adjusted = R_offset @ R_apriltag
                    rvec_adjusted = cv2.Rodrigues(R_adjusted)[0]
                    
                    # 平移调整
                    tvec_adjusted = apriltag_tvec * trans_scale
                    
                    success, rvec, tvec = cv2.solvePnP(
                        objpoints, imgpoints, camera_matrix, dist_coeffs,
                        rvec=rvec_adjusted,
                        tvec=tvec_adjusted,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    if success:
                        error = self._calculate_reprojection_error(
                            rvec, tvec, objpoints, imgpoints, camera_matrix, dist_coeffs
                        )
                        
                        # 如果误差足够小，返回这个解
                        if error < self.max_reprojection_error:
                            return ('CORRECTED_APRILTAG', rvec, tvec, error)
        
        except Exception as e:
            logger.debug(f"修正AprilTag约束方法失败: {e}")
        
        return None
    
    def _calculate_reprojection_error(self,
                                      rvec: np.ndarray,
                                      tvec: np.ndarray,
                                      objpoints: np.ndarray,
                                      imgpoints: np.ndarray,
                                      camera_matrix: np.ndarray,
                                      dist_coeffs: np.ndarray) -> float:
        """计算重投影误差"""
        try:
            projected_points, _ = cv2.projectPoints(
                objpoints, rvec, tvec, camera_matrix, dist_coeffs
            )
            
            errors = np.linalg.norm(
                projected_points.reshape(-1, 2) - imgpoints.reshape(-1, 2),
                axis=1
            )
            
            return np.mean(errors)
        except:
            return float('inf')
    
    def _check_apriltag_consistency(self,
                                    rvec: np.ndarray,
                                    tvec: np.ndarray,
                                    apriltag_rvec: np.ndarray,
                                    apriltag_tvec: np.ndarray) -> Dict:
        """检查与AprilTag位姿的一致性"""
        try:
            # 旋转角度差异
            R1 = cv2.Rodrigues(rvec)[0]
            R2 = cv2.Rodrigues(apriltag_rvec)[0]
            R_diff = R1.T @ R2
            angle_diff = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
            
            # 平移距离差异
            trans_diff = np.linalg.norm(tvec - apriltag_tvec)
            
            return {
                'angle_difference_deg': angle_diff,
                'translation_difference_mm': trans_diff,
                'is_consistent': angle_diff < self.angle_consistency_threshold
            }
        except:
            return {'error': 'Failed to check consistency'}
    
    def _select_best_solution(self,
                              solutions: List[Tuple],
                              objpoints: np.ndarray,
                              imgpoints: np.ndarray,
                              camera_matrix: np.ndarray,
                              dist_coeffs: np.ndarray,
                              apriltag_rvec: np.ndarray,
                              apriltag_tvec: np.ndarray) -> Optional[Tuple]:
        """选择最佳解"""
        if not solutions:
            return None
        
        # 过滤掉重投影误差过大的解
        valid_solutions = [s for s in solutions if s[3] < self.max_reprojection_error]
        
        if not valid_solutions:
            # 如果没有满足阈值的解，选择误差最小的
            logger.warning(f"所有解的重投影误差都超过阈值 {self.max_reprojection_error}px")
            return min(solutions, key=lambda x: x[3])
        
        # 在有效解中，优先选择与AprilTag一致性好的
        scored_solutions = []
        for solution in valid_solutions:
            method, rvec, tvec, error = solution
            consistency = self._check_apriltag_consistency(
                rvec, tvec, apriltag_rvec, apriltag_tvec
            )
            
            # 计算综合得分（误差越小越好，一致性越好越好）
            if 'angle_difference_deg' in consistency:
                angle_penalty = consistency['angle_difference_deg'] / 180.0  # 归一化到[0,1]
                score = error + angle_penalty * 50  # 角度不一致性的惩罚
            else:
                score = error
            
            scored_solutions.append((score, solution))
        
        # 选择得分最低的解
        best_score, best_solution = min(scored_solutions, key=lambda x: x[0])
        
        logger.info(f"选择最佳PnP解: {best_solution[0]}, 误差: {best_solution[3]:.3f}px")
        
        return best_solution


def test_pnp_ambiguity_resolver():
    """测试PnP多解歧义解决器"""
    resolver = PnPAmbiguityResolver()
    print("PnP多解歧义解决器初始化完成")
    print(f"最大重投影误差阈值: {resolver.max_reprojection_error}px")
    print(f"角度一致性阈值: {resolver.angle_consistency_threshold}°")


if __name__ == '__main__':
    test_pnp_ambiguity_resolver()