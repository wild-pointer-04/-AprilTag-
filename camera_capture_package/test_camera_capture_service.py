#!/usr/bin/env python3
"""
相机拍照服务测试脚本

功能：
1. 测试拍照服务是否正常工作
2. 模拟机械臂多点拍照流程
3. 验证服务响应

使用方法:
    # 单次拍照测试
    python test_camera_capture_service.py --single
    
    # 多次拍照测试（模拟机械臂工作流程）
    python test_camera_capture_service.py --count 5 --interval 2.0
    
    # 批量拍照测试
    python test_camera_capture_service.py --batch 20
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import argparse
import time
import sys


class CaptureServiceTester(Node):
    """拍照服务测试节点"""
    
    def __init__(self, service_name: str = '/camera_capture'):
        super().__init__('capture_service_tester')
        self.service_name = service_name
        
        # 创建服务客户端
        self.capture_client = self.create_client(Trigger, service_name)
        
        self.get_logger().info(f'正在连接服务: {service_name}')
    
    def wait_for_service(self, timeout_sec: float = 10.0) -> bool:
        """等待服务可用"""
        self.get_logger().info('等待拍照服务...')
        
        start_time = time.time()
        while not self.capture_client.wait_for_service(timeout_sec=1.0):
            if time.time() - start_time > timeout_sec:
                self.get_logger().error(f'服务 {self.service_name} 在 {timeout_sec} 秒内未响应')
                return False
            self.get_logger().info('服务尚未就绪，继续等待...')
        
        self.get_logger().info('✅ 服务已就绪')
        return True
    
    def capture_photo(self) -> tuple:
        """
        调用拍照服务
        
        返回:
            (success, message): 成功标志和响应消息
        """
        request = Trigger.Request()
        
        try:
            future = self.capture_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            if future.done():
                response = future.result()
                return response.success, response.message
            else:
                self.get_logger().error('服务调用超时')
                return False, '服务调用超时'
                
        except Exception as e:
            self.get_logger().error(f'服务调用异常: {e}')
            return False, str(e)
    
    def test_single_capture(self):
        """测试单次拍照"""
        self.get_logger().info('='*60)
        self.get_logger().info('开始单次拍照测试')
        self.get_logger().info('='*60)
        
        if not self.wait_for_service():
            return False
        
        self.get_logger().info('📸 正在拍照...')
        success, message = self.capture_photo()
        
        if success:
            self.get_logger().info(f'✅ 拍照成功: {message}')
            return True
        else:
            self.get_logger().error(f'❌ 拍照失败: {message}')
            return False
    
    def test_multiple_captures(self, count: int, interval: float):
        """
        测试多次拍照（模拟机械臂工作流程）
        
        参数:
            count: 拍照次数
            interval: 拍照间隔（秒）
        """
        self.get_logger().info('='*60)
        self.get_logger().info(f'开始多次拍照测试（共 {count} 次，间隔 {interval} 秒）')
        self.get_logger().info('='*60)
        
        if not self.wait_for_service():
            return False
        
        success_count = 0
        failure_count = 0
        
        for i in range(count):
            self.get_logger().info(f'\n--- 第 {i+1}/{count} 次拍照 ---')
            
            # 模拟机械臂移动
            self.get_logger().info(f'🤖 模拟机械臂移动到位置 {i+1}...')
            time.sleep(0.5)  # 模拟移动时间
            
            # 拍照
            self.get_logger().info('📸 正在拍照...')
            success, message = self.capture_photo()
            
            if success:
                self.get_logger().info(f'✅ {message}')
                success_count += 1
            else:
                self.get_logger().error(f'❌ 拍照失败: {message}')
                failure_count += 1
            
            # 等待间隔
            if i < count - 1:
                self.get_logger().info(f'等待 {interval} 秒...')
                time.sleep(interval)
        
        # 统计结果
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('测试完成')
        self.get_logger().info(f'  总次数: {count}')
        self.get_logger().info(f'  成功: {success_count}')
        self.get_logger().info(f'  失败: {failure_count}')
        self.get_logger().info(f'  成功率: {success_count/count*100:.1f}%')
        self.get_logger().info('='*60)
        
        return failure_count == 0
    
    def test_batch_capture(self, count: int):
        """
        批量快速拍照测试
        
        参数:
            count: 拍照次数
        """
        self.get_logger().info('='*60)
        self.get_logger().info(f'开始批量拍照测试（共 {count} 次）')
        self.get_logger().info('='*60)
        
        if not self.wait_for_service():
            return False
        
        success_count = 0
        failure_count = 0
        start_time = time.time()
        
        for i in range(count):
            success, message = self.capture_photo()
            
            if success:
                success_count += 1
                self.get_logger().info(f'[{i+1}/{count}] ✅ {message}')
            else:
                failure_count += 1
                self.get_logger().error(f'[{i+1}/{count}] ❌ 失败: {message}')
        
        elapsed_time = time.time() - start_time
        
        # 统计结果
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('批量测试完成')
        self.get_logger().info(f'  总次数: {count}')
        self.get_logger().info(f'  成功: {success_count}')
        self.get_logger().info(f'  失败: {failure_count}')
        self.get_logger().info(f'  成功率: {success_count/count*100:.1f}%')
        self.get_logger().info(f'  总耗时: {elapsed_time:.2f} 秒')
        self.get_logger().info(f'  平均速度: {count/elapsed_time:.2f} 张/秒')
        self.get_logger().info('='*60)
        
        return failure_count == 0


def main(args=None):
    """主函数"""
    parser = argparse.ArgumentParser(description='相机拍照服务测试脚本')
    parser.add_argument('--service-name', type=str, default='/camera_capture',
                       help='服务名称（默认：/camera_capture）')
    parser.add_argument('--single', action='store_true',
                       help='单次拍照测试')
    parser.add_argument('--count', type=int, default=5,
                       help='多次拍照测试的次数（默认：5）')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='多次拍照测试的间隔秒数（默认：2.0）')
    parser.add_argument('--batch', type=int, default=None,
                       help='批量快速拍照测试的次数')
    
    # 解析参数
    if args is None:
        cli_args, _ = parser.parse_known_args()
    else:
        cli_args, _ = parser.parse_known_args(args)
    
    # 初始化 ROS2
    rclpy.init(args=args)
    
    # 创建测试节点
    tester = CaptureServiceTester(service_name=cli_args.service_name)
    
    try:
        # 根据参数选择测试模式
        if cli_args.single:
            # 单次拍照测试
            success = tester.test_single_capture()
        elif cli_args.batch is not None:
            # 批量拍照测试
            success = tester.test_batch_capture(cli_args.batch)
        else:
            # 多次拍照测试（默认）
            success = tester.test_multiple_captures(cli_args.count, cli_args.interval)
        
        # 返回退出码
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        tester.get_logger().info('接收到中断信号，正在退出...')
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
