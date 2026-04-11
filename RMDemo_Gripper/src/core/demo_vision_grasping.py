#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
# ============================================================
# vi_grab_final.py
# 基于 requirements.md 完整实现的抓取流程
# 功能：满足 requirements.md 中的所有要求
# ============================================================
import sys
import os
import time
import argparse
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from densefusion_ros.msg import PoseEstimationResult
    # from onepose_ros_demo.msg import PoseEstimationResult
    ROS2_POSE_MSG_AVAILABLE = True
except ImportError:
    ROS2_POSE_MSG_AVAILABLE = False

# 添加项目路径R
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
sys.path.append(project_root)

try:
    # 尝试导入 RM API
    # Add the parent directory of src to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    from src.Robotic_Arm.rm_robot_interface import *
    print("成功导入 RM API 接口")
except ImportError:
    raise RuntimeError("无法导入 RM API，程序仅支持真实机械臂模式")


class VisionGraspingSystem:
    """
    视觉抓取系统主类
    实现 requirements.md 中的所有功能需求
    """
    
    def __init__(
        self,
        ip="192.168.1.18",
        port=8080,
        pose_topic="/pose_estimation_result",
    ):
        """
        初始化视觉抓取系统
        
        Args:
            ip: 机械臂 IP 地址
            port: 机械臂端口
        """
        self.ip = ip
        self.port = port
        
        # ── 手眼标定参数 ──────────────────────────────────────────────────────────
        # 相机坐标系到机械臂末端坐标系的旋转矩阵，通过手眼标定得到( yaw-pitch-roll = -0.5, 2.0, -89)
        self.rotation_matrix = np.array([[0.01206237, 0.99929647, 0.03551135],
                                        [-0.99988374, 0.01172294, 0.00975125],
                                        [0.00932809, -0.03562485, 0.9993217]])
        # self.rotation_matrix = np.eye(3, dtype=float)
        # 相机坐标系到机械臂末端坐标系的平移向量，通过手眼标定得到（单位：m）
        # 原始数据:
        # self.translation_vector = np.array([-0.08039019, 0.03225555, -0.08256825])
        # self.translation_vector = np.array([-0.0, 0.0, -0.0])
        # 待标定数据
        self.translation_vector = np.array([-0.14039019, -0.05225555, -0.12256825])

        # PoseEstimationResult(x左,y上,z后) -> 末端执行器(x上,y右,z前) 轴映射
        self.pose_result_to_ee_axes = np.array([
            [0.0, 1.0, 0.0],    # x_ee = y_pose
            [-1.0, 0.0, 0.0],   # y_ee = -x_pose
            [0.0, 0.0, 1.0],   # z_ee = z_pose
        ], dtype=float)
        # self.pose_result_to_ee_axes = np.eye(3, dtype=float)
        
        # ── 状态机变量 ────────────────────────────────────────────────────────────
        self.current_state = "INIT"  # 状态机状态
        self.retry_count = 0
        self.max_retries = 3
        self.object_detected = False
        self.object_pose_in_base = None  # 物体在基坐标系下的位姿
        self.detection_attempts = 0
        self.max_detection_attempts = 5
        self.state_failures = {}
        self.max_state_retries = 2
        self.state_retry_limits = {
            "INIT": 2,
            "MOVE_TO_HOME": 2,
            "MOVE_TO_DETECTION_POSE": 2,
            "CHECK_OBJECT": 2,
            "MOVE_TO_PRE_GRASP": 2,
            "OPEN_GRIPPER": 2,
            "MOVE_TO_GRASP": 2,
            "CLOSE_GRIPPER": 2,
            "CHECK_GRASP_SUCCESS": 1,
            "PLACE_OBJECT": 2,
            "RETURN_TO_DETECTION_POSE": 2,
            "RETURN_TO_HOME": 2,
        }
        self.last_error = ""
        self.safe_stop_reason = ""
        self.recovery_target_state = "CHECK_OBJECT"

        # ── ROS2 PoseEstimationResult 订阅状态（DenseFusion）──────────────────────
        self.pose_topic = str(pose_topic)
        self.ros_node = None
        self.pose_subscriber = None
        self.pose_queue_maxlen = 1
        self.pose_msg_queue = deque()
        self.latest_pose_msg = None
        self.latest_pose_msg_time = 0.0
        self.use_topic_pose = False
        self._active_test_frame_id = None
        self._verification_records = []
        
        # ── 真实机械臂 + DenseFusion topic 初始化（唯一模式）──────────────────────
        self._init_robot_arm()
        self._init_pose_subscriber()
        
        # ── 初始位姿和检测位姿 ────────────────────────────────────────────────
        self.home_joint_angles = [0.298, 26.316, -134.891, 2.141, -2.522, 3.098]  # home mode
        self.detection_joint_angles = [-0.144, 8.962, -145.927, -2.342, 44.379, 2.226]  # detect mode
        self.ready_joint_angle = [0.298, 26.316, -134.891, 2.141, -2.522, 3.098]  # ready mode 
        # self.pre_grab_joint_angle = [0.307, -1.470, -150.770, 2.131, 62.021, 3.337]  # pre grab mode
        
        print("视觉抓取系统初始化完成")

    def _init_pose_subscriber(self):
        """初始化 PoseEstimationResult 订阅（必需）"""
        if not ROS2_POSE_MSG_AVAILABLE:
            raise RuntimeError("[DenseFusion] 缺少 rclpy 或 densefusion_ros.msg，无法运行")

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.ros_node = Node("rm_vision_grasping_pose_listener")
            latest_only_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=self.pose_queue_maxlen,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            self.pose_subscriber = self.ros_node.create_subscription(
                PoseEstimationResult,
                self.pose_topic,
                self._pose_result_callback,
                latest_only_qos,
            )
            self.use_topic_pose = True
            print(f"[DenseFusion] 已订阅话题: {self.pose_topic}")
        except Exception as e:
            raise RuntimeError(f"[DenseFusion] 订阅初始化失败: {e}")

    def _pose_result_callback(self, msg):
        """保存 PoseEstimationResult（仅保留最新一帧）"""
        msg_time = time.time()
        self.latest_pose_msg = msg
        self.latest_pose_msg_time = msg_time

        # 仅保留最新消息，避免消费到历史积压数据。
        self.pose_msg_queue.clear()
        self.pose_msg_queue.append((msg_time, msg))

    @staticmethod
    def _normalize_pose_translation_m(translation_vec):
        """
        将 PoseEstimationResult 平移向量按“米”解析。
        约定: 上游 PoseEstimationResult.translation_vector 单位为 m。
        """
        t_m = np.asarray(translation_vec, dtype=float)
        # 米制下，机械臂近场抓取通常不会超过数米；超范围给出提示，便于排查上游单位配置。
        if np.max(np.abs(t_m)) > 1.0:
            print(
                "[DenseFusion][单位提示] translation_vector 数值较大，当前代码按米(m)解析；"
                "如上游输出为毫米(mm)，请在上游统一改为米。"
            )
        return t_m

    def _convert_pose_result_vector_to_ee(self, vec3):
        """PoseEstimationResult 坐标系向量 -> 末端执行器坐标系向量"""
        return self.pose_result_to_ee_axes.dot(np.asarray(vec3, dtype=float))

    def _convert_pose_result_rotation_to_ee(self, rotation_matrix):
        """PoseEstimationResult 坐标系旋转矩阵 -> 末端执行器坐标系旋转矩阵"""
        c = self.pose_result_to_ee_axes
        return c.dot(rotation_matrix).dot(c.T)

    @staticmethod
    def _log_pose_xyzrpy(tag, pose):
        """统一打印位姿: 偏移向量xyz + 欧拉角rpy"""
        rpy_deg = np.degrees(np.asarray(pose[3:6], dtype=float))
        print(
            f"[{tag}] xyz=[{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}] m, "
            f"rpy=[{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}] deg"
        )

    def get_pose_from_topic(
        self,
        timeout_sec=3.0,
        require_fresh=True,
        max_msg_age_sec=1.0,
        log_timeout=True,
    ):
        """
        从 PoseEstimationResult 话题获取一帧位姿。
        返回: [rx, ry, rz, tx, ty, tz]（均在末端坐标系）或 None
        """
        if not self.use_topic_pose or self.ros_node is None:
            return None

        start_time = time.time()
        # 仅接受本次调用之后收到的新消息，避免重复消费旧缓存消息
        request_time = start_time
        while time.time() - start_time < timeout_sec:
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)
            latest_valid_msg = self.latest_pose_msg
            latest_valid_msg_time = self.latest_pose_msg_time
            if latest_valid_msg is not None:
                # 过滤旧消息或过期消息，避免抓取后误判“仍检测到物体”
                if require_fresh and latest_valid_msg_time < request_time:
                    continue
                if require_fresh and (time.time() - latest_valid_msg_time) > max_msg_age_sec:
                    continue
                if len(latest_valid_msg.rotation_matrix) != 9 or len(latest_valid_msg.translation_vector) != 3:
                    print("[DenseFusion] 消息字段长度异常，忽略本帧")
                    continue
                self.latest_pose_msg_time = latest_valid_msg_time
                # 物体在相机坐标系下的旋转矩阵和平移向量
                r_pose = np.array(latest_valid_msg.rotation_matrix, dtype=float).reshape(3, 3)
                t_pose_m = self._normalize_pose_translation_m(latest_valid_msg.translation_vector)
                # 末端运动方向 x_ee = -x_cam
                # t_pose_m[0] = -t_pose_m[0]

                # r_pose = self._convert_pose_result_rotation_to_ee(r_pose)
                # t_pose = self._convert_pose_result_vector_to_ee(t_pose)
                rvec_camera = R.from_matrix(r_pose).as_rotvec()
                euler_camera = R.from_matrix(r_pose).as_euler('xyz', degrees=False)
                euler_camera_deg = np.degrees(euler_camera)
                print("[DenseFusion] 收到检测结果（相机坐标系下的物体位姿）")
                print(f"  offset(xyz): [{t_pose_m[0]:.4f}, {t_pose_m[1]:.4f}, {t_pose_m[2]:.4f}] m")
                print(
                    f"  euler_xyz: [{euler_camera[0]:.4f}, "
                    f"{euler_camera[1]:.4f}, {euler_camera[2]:.4f}] rad"
                )
                print(
                    f"  euler_xyz: [{euler_camera_deg[0]:.2f}, "
                    f"{euler_camera_deg[1]:.2f}, {euler_camera_deg[2]:.2f}] deg"
                )
                print(f"  rvec_camera: [{rvec_camera[0]:.4f}, {rvec_camera[1]:.4f}, {rvec_camera[2]:.4f}] rad")
                return [rvec_camera[0], rvec_camera[1], rvec_camera[2], t_pose_m[0], t_pose_m[1], t_pose_m[2]]

        if log_timeout:
            print(f"[DenseFusion] 超时未收到新鲜消息: {self.pose_topic}")
        return None

    def check_no_fresh_pose_stable(self, required_no_msg=3, per_check_timeout=1.0):
        """
        抗抖判断：必须连续 required_no_msg 次未收到新鲜消息，才认为目标已消失（抓取成功）。
        返回:
          - True: 连续无新消息，判定抓取成功
          - False: 检查过程中收到新消息，判定物体仍在
          - None: 未启用topic订阅，无法使用该策略
        """
        if not self.use_topic_pose or self.ros_node is None:
            return None

        no_msg_count = 0
        for i in range(required_no_msg):
            pose = self.get_pose_from_topic(
                timeout_sec=per_check_timeout,
                require_fresh=True,
                max_msg_age_sec=1.0,
                log_timeout=False,
            )
            if pose is None:
                no_msg_count += 1
                print(
                    f"[抗抖检查] 第 {i + 1}/{required_no_msg} 次: 未收到新消息"
                )
            else:
                print(
                    f"[抗抖检查] 第 {i + 1}/{required_no_msg} 次: 收到新消息，判定物体仍在"
                )
                return False

        return no_msg_count >= required_no_msg
    
    def _init_robot_arm(self):
        """初始化真实机械臂"""
        try:
            self.thread_mode = rm_thread_mode_e(2)  # 三线程模式
            self.robot = RoboticArm(self.thread_mode)
            self.handle = self.robot.rm_create_robot_arm(self.ip, self.port, 3)
            
            if self.handle.id == -1:
                raise ConnectionError("连接机械臂失败")
            else:
                print(f"成功连接到机械臂: {self.handle.id}")
                
        except Exception as e:
            raise RuntimeError(f"初始化机械臂失败: {e}")

    @staticmethod
    def _pad_joint_degree_7(joint_deg):
        """将关节角列表补齐/截断到7维（单位°）"""
        vals = [float(v) for v in joint_deg]
        if len(vals) < 7:
            vals.extend([0.0] * (7 - len(vals)))
        return vals[:7]

    def _plan_movej_p_joint_solution(self, pose):
        """
        对目标位姿做逆解，返回 (ok, q_solve_deg, message)
        """
        current_joint = self.get_current_joint_angles()
        if current_joint is None:
            return False, None, "无法获取当前关节角，逆解规划失败"

        try:
            ik_params = rm_inverse_kinematics_params_t(
                q_in=self._pad_joint_degree_7(current_joint),
                q_pose=[float(v) for v in pose[:6]],
                flag=1,  # 欧拉角
            )
            ik_ret, q_solve = self.robot.rm_algo_inverse_kinematics(ik_params)
        except Exception as e:
            return False, None, f"逆解异常: {e}"

        if ik_ret != 0:
            return False, None, f"逆解失败, ret={ik_ret}"

        return True, [float(v) for v in q_solve], "逆解成功"

    def _check_movej_p_self_collision(self, pose):
        """
        在下发 movej_p 前做自碰撞前置检查。
        返回 (safe, reason)
        """

        ok, q_solve, msg = self._plan_movej_p_joint_solution(pose)
        if not ok:
            return False, msg
        # if q_solve[4] < -66.0 or q_solve[4] > 115.0:
        #     return False, "关节5超出限位, Limit: [-66, 115]"
        # 六轴机型可做关节限位检查（返回>0表示第i个关节超限）
        if len(q_solve) >= 6:
            try:
                limit_ret = self.robot.rm_algo_ikine_check_joint_position_limit(
                    q_solve[:6]
                )
                if limit_ret not in (0, -1):
                    return False, f"关节限位检查失败, joint={limit_ret}"
            except Exception:
                # 非六轴或底层不支持时忽略该项
                pass

        # 自碰撞检测：0无碰撞，1碰撞/超限
        collision_ret = self.robot.rm_algo_safety_robot_self_collision_detection(
            self._pad_joint_degree_7(q_solve)
        )
        if collision_ret != 0:
            return False, f"自碰撞检测失败, ret={collision_ret}"

        return True, "自碰撞检查通过"
    
    def movej(self, joint_angles, v=30, r=0, connect=0, block=1):
        """
        关节空间运动
        
        Args:
            joint_angles: 关节角度（弧度）
            v: 速度百分比
            r: 融合半径
            connect: 轨迹连接标志
            block: 是否阻塞
        """
        print(f"[MoveJ] 移动到关节角度: {joint_angles}")
        
        movej_result = self.robot.rm_movej(joint_angles, v, r, connect, block)
        if movej_result == 0:
            print("关节运动成功")
            ok = True
        else:
            print(f"关节运动失败, 错误码: {movej_result}")
            ok = False
        time.sleep(2)
        return ok
    
    def movej_p(self, pose, v=30, r=0, connect=0, block=1):
        """
        笛卡尔空间关节运动
        
        Args:
            pose: 位姿 [x, y, z, rx, ry, rz]（米和弧度）
            v: 速度百分比
            r: 融合半径
            connect: 轨迹连接标志
            block: 是否阻塞
        """
        print(f"[MoveJ_P] 移动到笛卡尔位姿: {pose[:3]}")
        self._log_pose_xyzrpy("机械臂下发(movej_p)", pose)
        
        safe, reason = self._check_movej_p_self_collision(pose)
        if not safe:
            print(f"[MoveJ_P] 已拦截，未下发运动: {reason}")
            return False

        movej_p_result = self.robot.rm_movej_p(pose, v, r, connect, block)
        if movej_p_result == 0:
            print("笛卡尔关节运动成功")
            ok = True
        else:
            print(f"笛卡尔关节运动失败, 错误码: {movej_p_result}")
            ok = False
        time.sleep(2)
        return ok
    
    def movel(self, pose, v=30, r=0, connect=0, block=1):
        """
        笛卡尔空间直线运动
        
        Args:
            pose: 位姿 [x, y, z, rx, ry, rz]（米和弧度）
            v: 速度百分比
            r: 融合半径
            connect: 轨迹连接标志
            block: 是否阻塞
        """
        print(f"[MoveL] 直线运动到位姿: {pose[:3]}")
        self._log_pose_xyzrpy("机械臂下发(movel)", pose)
        
        movel_result = self.robot.rm_movel(pose, v, r, connect, block)
        if movel_result == 0:
            print("直线运动成功")
            ok = True
        else:
            print(f"直线运动失败, 错误码: {movel_result}")
            ok = False
        time.sleep(2)
        return ok
    
    def set_gripper_pick_on(self, speed=500, force=200, block=True, timeout=30):
        """夹爪力控抓取"""
        print("[Gripper] 执行夹爪力控抓取")
        try:
            gripper_result = self.robot.rm_set_gripper_pick_on(speed, force, block, timeout)
            if gripper_result == 0:
                print("夹爪抓取成功")
                ok = True
            else:
                print(f"夹爪抓取失败, 错误码: {gripper_result}")
                ok = False
        except Exception as e:
            print(f"夹爪抓取异常: {e}")
            ok = False

        time.sleep(1)
        return ok
    
    def set_gripper_release(self, speed=500, block=True, timeout=30):
        """夹爪释放"""
        print("[Gripper] 执行夹爪释放")
        try:
            gripper_result = self.robot.rm_set_gripper_release(speed, block, timeout)
            if gripper_result == 0:
                print("夹爪释放成功")
                ok = True
            else:
                print(f"夹爪释放失败, 错误码: {gripper_result}")
                ok = False
        except Exception as e:
            print(f"夹爪释放异常: {e}")
            ok = False

        time.sleep(1)
        return ok
    
    def get_current_joint_angles(self):
        """获取当前关节角度"""
        ret_code, joint_degrees = self.robot.rm_get_joint_degree()
        if ret_code == 0:
            return joint_degrees
        print(f"获取关节角度失败, 错误码: {ret_code}")
        return None
    
    def get_current_cartesian_pose(self):
        """获取当前笛卡尔位姿"""
        joint_angles = self.get_current_joint_angles()
        if joint_angles is None:
            return None
        pose = self.robot.rm_algo_forward_kinematics(joint_angles, flag=1)
        return pose
    
    # 相机坐标系物体到机械臂基坐标系转换函数
    def convert(
        self,
        x,
        y,
        z,
        x1,
        y1,
        z1,
        rx,
        ry,
        rz,
        obj_rotation_matrix=None,
    ):
        """
        我们需要将旋转向量和平移向量转换为齐次变换矩阵，然后使用深度相机识别到的物体坐标（x, y, z）和
        机械臂末端的位姿（x1,y1,z1,rx,ry,rz）来计算物体相对于机械臂基座的位姿（x, y, z, rx, ry, rz）
        """
        # # 相机坐标系到机械臂末端坐标系的旋转矩阵和平移向量
        # rotation_matrix = np.array([[ 0.01206237 , 0.99929647  ,0.03551135],
        #                             [-0.99988374 , 0.01172294 , 0.00975125],
        #                             [ 0.00932809 ,-0.03562485 , 0.9993217 ]])
        # # rotation_matrix = np.eye(3, dtype=float)
        # translation_vector = np.array([-0.06039019, 0.03225555, -0.06256825])
        rotation_matrix = self.rotation_matrix
        translation_vector = self.translation_vector
        # 深度相机识别物体返回的坐标
        # 和机械臂末端的X轴,y轴反向
        x = x * -1.0
        # y = y + 0.16
        obj_camera_coordinates = np.array([x, y, z])

        # 机械臂末端的位姿，单位为弧度
        end_effector_pose = np.array([x1, y1, z1,
                                    rx, ry, rz])
        # 将旋转矩阵和平移向量转换为齐次变换矩阵
        T_camera_to_end_effector = np.eye(4)
        T_camera_to_end_effector[:3, :3] = rotation_matrix
        T_camera_to_end_effector[:3, 3] = translation_vector
        # 机械臂末端的位姿转换为齐次变换矩阵
        position = end_effector_pose[:3]
        orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
        T_base_to_end_effector = np.eye(4)
        T_base_to_end_effector[:3, :3] = orientation
        T_base_to_end_effector[:3, 3] = position
        # 计算物体相对于机械臂基座的位姿
        obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1])  # 将物体坐标转换为齐次坐标
        #obj_end_effector_coordinates_homo = np.linalg.inv(T_camera_to_end_effector).dot(obj_camera_coordinates_homo)
        obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(obj_camera_coordinates_homo)
        obj_base_coordinates_homo = T_base_to_end_effector.dot(obj_end_effector_coordinates_homo)
        obj_base_coordinates = obj_base_coordinates_homo[:3]  # 从齐次坐标中提取物体的x, y, z坐标
        # 计算物体旋转：R_base_obj = R_base_ee * R_ee_camera * R_camera_obj
        # 若未提供物体在相机坐标系下旋转，则按单位阵退化（仅使用坐标链路）
        if obj_rotation_matrix is None:
            obj_rotation_matrix = np.eye(3, dtype=float)
        else:
            obj_rotation_matrix = np.asarray(obj_rotation_matrix, dtype=float).reshape(3, 3)
        obj_orientation_matrix = (
            T_base_to_end_effector[:3, :3]
            .dot(rotation_matrix)
            .dot(obj_rotation_matrix)
        )
        obj_orientation_euler = R.from_matrix(obj_orientation_matrix).as_euler('xyz', degrees=False)
        obj_orientation_euler_deg = np.degrees(obj_orientation_euler)
        print("[旋转调试] R_camera_obj:")
        print(np.array2string(obj_rotation_matrix, precision=6, suppress_small=True))
        print("[旋转调试] R_base_obj:")
        print(np.array2string(obj_orientation_matrix, precision=6, suppress_small=True))
        print(
            "[旋转调试] obj_orientation_euler(rad): "
            f"[{obj_orientation_euler[0]:.6f}, {obj_orientation_euler[1]:.6f}, {obj_orientation_euler[2]:.6f}]"
        )
        print(
            "[旋转调试] obj_orientation_euler(deg): "
            f"[{obj_orientation_euler_deg[0]:.3f}, {obj_orientation_euler_deg[1]:.3f}, {obj_orientation_euler_deg[2]:.3f}]"
        )
        # 组合结果
        obj_base_pose = np.hstack((obj_base_coordinates, obj_orientation_euler))
        obj_base_pose[3:] = rx,ry,rz
        
        return obj_base_pose

    def convert_camera_to_base(self, obj_camera_pose, input_frame="camera"):
        """
        将相机坐标系下的物体位姿转换到机械臂基坐标系
        
        Args:
            obj_camera_pose: [rx, ry, rz, tx, ty, tz]
                rx, ry, rz: Rodrigues 旋转向量（弧度）
                tx, ty, tz: 平移向量（米）
            input_frame: 输入位姿坐标系
                - "camera": 相机坐标系（使用手眼标定 T_camera->ee）
                - "end_effector": 末端坐标系（直接作为 ee 下位姿）
        
        Returns:
            list: [x, y, z, rx, ry, rz]，物体在机械臂基坐标系下的位姿
        """
        # 解析输入（兼容两种格式）:
        # 1) [rx, ry, rz, tx, ty, tz] 旋转向量+平移
        # 2) [tx, ty, tz, qx, qy, qz, qw] 平移+四元数
        if len(obj_camera_pose) == 6:
            rvec_x, rvec_y, rvec_z = obj_camera_pose[0], obj_camera_pose[1], obj_camera_pose[2]
            tvec_x_m, tvec_y_m, tvec_z_m = obj_camera_pose[3], obj_camera_pose[4], obj_camera_pose[5]
            rvec_obj = np.array([rvec_x, rvec_y, rvec_z], dtype=float)
            R_in_o = R.from_rotvec(rvec_obj).as_matrix()
        elif len(obj_camera_pose) == 7:
            tvec_x_m, tvec_y_m, tvec_z_m = obj_camera_pose[0], obj_camera_pose[1], obj_camera_pose[2]
            qx, qy, qz, qw = obj_camera_pose[3], obj_camera_pose[4], obj_camera_pose[5], obj_camera_pose[6]
            R_in_o = R.from_quat([qx, qy, qz, qw]).as_matrix()
            rvec_obj = None
        else:
            raise ValueError(
                "obj_camera_pose 格式错误，需为6维(rotvec+trans)或7维(trans+quat)"
            )

        # 获取当前机械臂末端位姿
        end_effector_pose = self.get_current_cartesian_pose()
        if end_effector_pose is None:
            print("获取机械臂位姿失败")
            return None

        x1, y1, z1 = end_effector_pose[0], end_effector_pose[1], end_effector_pose[2]
        rx, ry, rz = end_effector_pose[3], end_effector_pose[4], end_effector_pose[5]

        # 将物体在相机坐标系下旋转传入 convert，正确计算 base 下姿态
        obj_base_pose = self.convert(
            tvec_x_m,
            tvec_y_m,
            tvec_z_m,
            x1,
            y1,
            z1,
            rx,
            ry,
            rz,
            obj_rotation_matrix=R_in_o,
        )
        return obj_base_pose.tolist()

        # obj_input_coordinates = np.array([tvec_x_m, tvec_y_m, tvec_z_m], dtype=float)

        # # ── 构建 T_base→end_effector（机械臂当前末端位姿）───────────────────
        # end_effector_position = np.array([x1, y1, z1])
        # end_effector_orientation = R.from_euler('xyz', [rx, ry, rz], degrees=False).as_matrix()
        # T_base_to_end_effector = np.eye(4)
        # T_base_to_end_effector[:3, :3] = end_effector_orientation
        # T_base_to_end_effector[:3, 3] = end_effector_position

        # # ── 计算物体在基坐标系下的位置 ───────────────────────────────────────
        # if input_frame == "camera":
        #     # 构建 T_camera→end_effector（手眼标定结果）
        #     T_camera_to_end_effector = np.eye(4)
        #     T_camera_to_end_effector[:3, :3] = self.rotation_matrix
        #     # T_camera_to_end_effector[:3, :3] = np.eye(3, dtype=float)
        #     T_camera_to_end_effector[:3, 3] = self.translation_vector

        #     obj_input_homo = np.append(obj_input_coordinates, 1.0)
        #     obj_end_effector_homo = T_camera_to_end_effector.dot(obj_input_homo)
        # elif input_frame == "end_effector":
        #     obj_end_effector_homo = np.append(obj_input_coordinates, 1.0)
        # else:
        #     raise ValueError(f"不支持的 input_frame: {input_frame}")

        # # 末端坐标系 → 基坐标系
        # obj_base_homo = T_base_to_end_effector.dot(obj_end_effector_homo)
        # obj_base_position = obj_base_homo[:3]

        # # ── 计算物体在基坐标系下的旋转 ───────────────────────────────────────
        # if rvec_obj is not None or len(obj_camera_pose) == 7:
        #     if input_frame == "camera":
        #         # 相机坐标系 -> 末端坐标系
        #         R_ee_o = self.rotation_matrix.dot(R_in_o)
        #     else:
        #         R_ee_o = R_in_o
        #     # 基坐标系下物体旋转矩阵
        #     obj_orientation_matrix = end_effector_orientation.dot(R_ee_o)
        #     obj_orientation_euler = R.from_matrix(obj_orientation_matrix).as_euler('xyz', degrees=False)
        # else:
        #     # 未提供 rvec，退化为仅使用标定旋转
        #     if input_frame == "camera":
        #         obj_orientation_matrix = end_effector_orientation.dot(self.rotation_matrix)
        #     else:
        #         obj_orientation_matrix = end_effector_orientation
        #     obj_orientation_euler = R.from_matrix(obj_orientation_matrix).as_euler('xyz', degrees=False)

        # # 组合结果（位置 + 欧拉角）
        # obj_base_pose = np.hstack((obj_base_position, obj_orientation_euler))
        # # 保持与原逻辑兼容：姿态覆盖为当前末端欧拉角
        # # obj_base_pose[3:] = rx, ry, rz
        
        # return obj_base_pose.tolist()
    
    def get_object_detection_pose(self):
        """
        仅从 DenseFusion PoseEstimationResult 话题获取检测结果。
        返回: (pose, frame)
          - pose: [rx, ry, rz, tx, ty, tz] 或 None
          - frame: "end_effector"
        """
        self.detection_attempts += 1
        topic_pose = self.get_pose_from_topic(timeout_sec=3.0)
        if topic_pose is not None:
            print(f"[Topic检测] 第 {self.detection_attempts} 次尝试：收到 {self.pose_topic} 消息")
            return topic_pose, "camera"
        print(f"[Topic检测] 第 {self.detection_attempts} 次尝试：未收到 {self.pose_topic} 消息")
        return None, "camera"

    def get_current_pose_from_arm_state(self):
        """
        使用 rm_get_current_arm_state 获取当前机械臂位姿，失败则回退到正运动学。
        返回 [x, y, z, rx, ry, rz] 或 None
        """
        try:
            ret, state = self.robot.rm_get_current_arm_state()
            if ret == 0 and isinstance(state, dict) and "pose" in state:
                pose = state["pose"]
                if isinstance(pose, list) and len(pose) >= 6:
                    return [float(v) for v in pose[:6]]
        except Exception as e:
            print(f"[状态查询] rm_get_current_arm_state 异常: {e}")
        return self.get_current_cartesian_pose()

    @staticmethod
    def _angle_diff(a, b):
        """将角度差约束到 [-pi, pi]"""
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi

    def verify_pose_reached(
        self,
        target_pose,
        stage="unknown",
        timeout_sec=5.0,
        pos_tol=0.03,
        rot_tol=0.35,
    ):
        """
        通过 API 回读确认是否到达目标位姿
        - pos_tol: 米
        - rot_tol: 弧度（欧拉角各轴最大绝对误差）
        """
        deadline = time.time() + timeout_sec
        last_pose = None
        while time.time() < deadline:
            current_pose = self.get_current_pose_from_arm_state()
            if current_pose is None:
                time.sleep(0.2)
                continue
            last_pose = current_pose
            pos_err = np.linalg.norm(
                np.array(current_pose[:3], dtype=float)
                - np.array(target_pose[:3], dtype=float)
            )
            rot_err_xyz = np.abs([
                self._angle_diff(current_pose[3], target_pose[3]),
                self._angle_diff(current_pose[4], target_pose[4]),
                self._angle_diff(current_pose[5], target_pose[5]),
            ])
            reached = bool(
                pos_err <= pos_tol and float(np.max(rot_err_xyz)) <= rot_tol
            )
            if reached:
                break
            time.sleep(0.2)
        else:
            reached = False
            pos_err = None
            rot_err_xyz = None

        record = {
            "test_frame_id": self._active_test_frame_id,
            "stage": stage,
            "reached": reached,
            "target_pose": [float(v) for v in target_pose[:6]],
            "actual_pose": [float(v) for v in last_pose[:6]] if last_pose else None,
            "position_error_m": float(pos_err) if pos_err is not None else None,
            "rotation_error_xyz_rad": (
                [float(v) for v in rot_err_xyz] if rot_err_xyz is not None else None
            ),
        }
        self._verification_records.append(record)
        print(
            f"[达位验证][{stage}] frame={self._active_test_frame_id}, "
            f"reached={reached}, pos_err={record['position_error_m']}"
        )
        return reached

    def run_dataset_verification(
        self,
        report_path,
        pre_grasp_lift=0.10,
    ):
        """
        保留接口但禁用离线数据模式。
        """
        raise RuntimeError("已移除离线测试数据模式，仅支持订阅DenseFusion topic模式")
    
    def check_object(self):
        """
        检测物体
        
        Returns:
            bool: 是否检测到物体
        """
        print("\n" + "="*60)
        print("开始检测物体")
        print("="*60)
        
        object_pose, pose_frame = self.get_object_detection_pose()
        
        if object_pose is not None:
            print(f"[物体检测] 检测到物体，位姿: {object_pose}")
            
            # 转换到基坐标系
            self.object_pose_in_base = self.convert_camera_to_base(
                object_pose,
                input_frame=pose_frame,
            )
            if self.object_pose_in_base is not None:
                base_rpy_deg = np.degrees(np.asarray(self.object_pose_in_base[3:6], dtype=float))
                print(f"[坐标转换] 基坐标系位姿:")
                print(f"  位置: [{self.object_pose_in_base[0]:.4f}, {self.object_pose_in_base[1]:.4f}, {self.object_pose_in_base[2]:.4f}] m")
                print(f"  姿态: [{self.object_pose_in_base[3]:.4f}, {self.object_pose_in_base[4]:.4f}, {self.object_pose_in_base[5]:.4f}] rad")
                print(f"  姿态: [{base_rpy_deg[0]:.2f}, {base_rpy_deg[1]:.2f}, {base_rpy_deg[2]:.2f}] deg")
                self.object_detected = True
                return True
            else:
                print("[坐标转换] 转换失败")
                self.object_detected = False
                return False
        else:
            print("[物体检测] 未检测到物体")
            self.object_detected = False
            return False
    
    def print_current_frame(self):
        """打印当前坐标系信息"""
        print("\n" + "="*60)
        print("打印当前坐标系信息")
        print("="*60)
        
        # 获取当前关节角度
        joint_angles = self.get_current_joint_angles()
        if joint_angles is not None:
            print(f"当前关节角度: {joint_angles}")
        
        # 获取当前笛卡尔位姿
        cartesian_pose = self.get_current_cartesian_pose()
        if cartesian_pose is not None:
            cart_rpy_deg = np.degrees(np.asarray(cartesian_pose[3:6], dtype=float))
            print(f"当前笛卡尔位姿:")
            print(f"  位置: [{cartesian_pose[0]:.4f}, {cartesian_pose[1]:.4f}, {cartesian_pose[2]:.4f}] m")
            print(f"  姿态: [{cartesian_pose[3]:.4f}, {cartesian_pose[4]:.4f}, {cartesian_pose[5]:.4f}] rad")
            print(f"  姿态: [{cart_rpy_deg[0]:.2f}, {cart_rpy_deg[1]:.2f}, {cart_rpy_deg[2]:.2f}] deg")

    def _mark_state_success(self, state_name):
        """状态执行成功后清理失败计数"""
        if state_name in self.state_failures:
            del self.state_failures[state_name]

    def _enter_failure(self, state_name, reason, recover_state="CHECK_OBJECT", max_retries=None):
        """统一失败处理入口：先进入恢复态，超过次数后进入安全停机"""
        if max_retries is not None:
            limit = max_retries
        else:
            limit = self.state_retry_limits.get(state_name, self.max_state_retries)
        fail_count = self.state_failures.get(state_name, 0) + 1
        self.state_failures[state_name] = fail_count
        self.last_error = f"{state_name}: {reason}"
        self.recovery_target_state = recover_state
        print(
            f"[失败处理] 状态 {state_name} 执行失败 "
            f"({fail_count}/{limit})，原因: {reason}"
        )
        if fail_count >= limit:
            self.safe_stop_reason = self.last_error
            print("[失败处理] 失败次数达到上限，进入 SAFE_STOP")
            self.current_state = "SAFE_STOP"
        else:
            self.current_state = "ERROR_RECOVERY"
    
    def  run_state_machine(self):
        """运行状态机"""
        print(f"\n当前状态: {self.current_state}")
        state_before_run = self.current_state

        try:
            if self.current_state == "INIT":
                print("\n" + "="*60)
                print("状态: INIT - 机械臂初始化设置")
                print("="*60)
                print("[初始化] 打开夹爪")
                if not self.set_gripper_release():
                    self._enter_failure("INIT", "初始化阶段夹爪打开失败", recover_state="INIT")
                    return
                self._mark_state_success("INIT")
                self.current_state = "MOVE_TO_HOME"

            elif self.current_state == "MOVE_TO_HOME":
                print("\n" + "="*60)
                print("状态: MOVE_TO_HOME - 移动到初始位姿")
                print("="*60)
                print("[运动] 移动到初始位姿")
                if not self.movej(self.home_joint_angles, v=30):
                    self._enter_failure("MOVE_TO_HOME", "机械臂未能到达初始位姿", recover_state="MOVE_TO_HOME")
                    return
                self._mark_state_success("MOVE_TO_HOME")
                self.current_state = "PRINT_FRAME"

            elif self.current_state == "PRINT_FRAME":
                print("\n" + "="*60)
                print("状态: PRINT_FRAME - 打印当前坐标系")
                print("="*60)
                self.print_current_frame()
                self._mark_state_success("PRINT_FRAME")
                self.current_state = "MOVE_TO_DETECTION_POSE"

            elif self.current_state == "MOVE_TO_DETECTION_POSE":
                print("\n" + "="*60)
                print("状态: MOVE_TO_DETECTION_POSE - 移动到检测物品位姿")
                print("="*60)
                print("[运动] 移动到检测物品位姿")
                if not self.movej(self.detection_joint_angles, v=30):
                    self._enter_failure(
                        "MOVE_TO_DETECTION_POSE",
                        "机械臂未能到达检测位姿",
                        recover_state="MOVE_TO_DETECTION_POSE",
                    )
                    return
                self._mark_state_success("MOVE_TO_DETECTION_POSE")
                self.current_state = "CHECK_OBJECT"

            elif self.current_state == "CHECK_OBJECT":
                print("\n" + "="*60)
                print("状态: CHECK_OBJECT - 检测物体")
                print("="*60)
                if self.check_object():
                    print("[物体检测] 检测到物体，开始抓取流程")
                    self.detection_attempts = 0
                    self._mark_state_success("CHECK_OBJECT")
                    self.current_state = "MOVE_TO_PRE_GRASP"
                else:
                    self.detection_attempts += 1
                    print(
                        f"[物体检测] 未检测到物体，等待3秒后重新检测 "
                        f"({self.detection_attempts}/{self.max_detection_attempts})"
                    )
                    if self.detection_attempts >= self.max_detection_attempts:
                        self._enter_failure(
                            "CHECK_OBJECT",
                            "连续多次未检测到物体",
                            recover_state="MOVE_TO_DETECTION_POSE",
                            max_retries=2,
                        )
                        return
                    time.sleep(3)

            elif self.current_state == "MOVE_TO_PRE_GRASP":
                print("\n" + "="*60)
                print("状态: MOVE_TO_PRE_GRASP - 移动到预抓取位姿")
                print("="*60)
                if self.object_pose_in_base is not None:
                    pre_grasp_pose = self.object_pose_in_base.copy()
                    # 预抓取位姿为base坐标系, 所以是x轴加10cm
                    pre_grasp_pose[0] += 0.15 # x轴加10cm

                    print(f"[运动] 移动到预抓取位姿")
                    print(f"  原始位姿: {self.object_pose_in_base[:3]}")
                    print(f"  预抓取位姿: {pre_grasp_pose[:3]}")
                    if not self.movel(pre_grasp_pose, v=20):
                        self._enter_failure(
                            "MOVE_TO_PRE_GRASP",
                            "机械臂未能到达预抓取位姿",
                            recover_state="CHECK_OBJECT",
                        )
                        return
                    self._mark_state_success("MOVE_TO_PRE_GRASP")
                    self.current_state = "OPEN_GRIPPER"
                else:
                    print("[错误] 物体位姿为空，返回检测状态")
                    self.current_state = "CHECK_OBJECT"

            elif self.current_state == "OPEN_GRIPPER":
                print("\n" + "="*60)
                print("状态: OPEN_GRIPPER - 打开夹爪")
                print("="*60)
                if not self.set_gripper_release():
                    self._enter_failure("OPEN_GRIPPER", "打开夹爪失败", recover_state="MOVE_TO_PRE_GRASP")
                    return
                self._mark_state_success("OPEN_GRIPPER")
                self.current_state = "MOVE_TO_GRASP"

            elif self.current_state == "MOVE_TO_GRASP":
                print("\n" + "="*60)
                print("状态: MOVE_TO_GRASP - 移动到抓取位姿")
                print("="*60)
                if self.object_pose_in_base is not None:
                    print(f"[运动] 移动到抓取位姿: {self.object_pose_in_base[:3]}")
                    if not self.movej_p(self.object_pose_in_base, v=30):
                        self._enter_failure("MOVE_TO_GRASP", "机械臂未能到达抓取位姿", recover_state="MOVE_TO_PRE_GRASP")
                        return
                    self._mark_state_success("MOVE_TO_GRASP")
                    self.current_state = "CLOSE_GRIPPER"
                else:
                    print("[错误] 物体位姿为空，返回检测状态")
                    self.current_state = "CHECK_OBJECT"

            elif self.current_state == "CLOSE_GRIPPER":
                print("\n" + "="*60)
                print("状态: CLOSE_GRIPPER - 关闭夹爪")
                print("="*60)
                if not self.set_gripper_pick_on():
                    self._enter_failure("CLOSE_GRIPPER", "夹爪闭合失败", recover_state="MOVE_TO_PRE_GRASP")
                    return
                self._mark_state_success("CLOSE_GRIPPER")
                self.current_state = "CHECK_GRASP_SUCCESS"

            elif self.current_state == "CHECK_GRASP_SUCCESS":
                # self.current_state = "PLACE_OBJECT"
                print("\n" + "="*60)
                print("状态: CHECK_GRASP_SUCCESS - 检查抓取是否成功")
                print("="*60)
                print("[运动] 返回到检测位姿")
                if not self.movej(self.detection_joint_angles, v=30):
                    self._enter_failure(
                        "CHECK_GRASP_SUCCESS",
                        "返回检测位姿失败，无法判定抓取结果",
                        recover_state="MOVE_TO_DETECTION_POSE",
                    )
                    return
                time.sleep(3)
                self.object_detected = False
                self.detection_attempts = 0
                print("[检查] 检查物体是否还在...")
                time.sleep(2)

                stable_absent = self.check_no_fresh_pose_stable(
                    required_no_msg=3,
                    per_check_timeout=1.0,
                )
                if stable_absent is True:
                    print("[检查] 连续3次无新消息，判定抓取成功！")
                    self.retry_count = 0
                    self._mark_state_success("CHECK_GRASP_SUCCESS")
                    self.current_state = "PLACE_OBJECT"
                    return
                if stable_absent is None:
                    self._enter_failure(
                        "CHECK_GRASP_SUCCESS",
                        "抓取结果不确定（未启用或无法使用抗抖检测）",
                        recover_state="MOVE_TO_DETECTION_POSE",
                        max_retries=1,
                    )
                    return

                self.retry_count += 1
                print(f"[重试机制] 抓取失败，物体仍在。重试次数: {self.retry_count}/{self.max_retries}")
                if self.retry_count >= self.max_retries:
                    print(f"[重试机制] 达到最大重试次数 {self.max_retries}，抓取失败，返回检测位姿")
                    self.retry_count = 0
                    self.current_state = "CHECK_OBJECT"
                else:
                    print(f"[重试机制] 进行第 {self.retry_count} 次重试：返回检测位姿并重新识别后再抓取")
                    self.current_state = "CHECK_OBJECT"

            elif self.current_state == "PLACE_OBJECT":
                print("\n" + "="*60)
                print("状态: PLACE_OBJECT - 抓取后放置")
                print("="*60)
                print("[运动] 移动到放置位姿...")
                current_pose = self.get_current_cartesian_pose()
                if current_pose is not None:
                    place_pose = current_pose.copy()
                    place_pose[1] += 0.2 # y轴向右移动20cm
                    place_pose[2] += 0.2 # x轴向上移动20cm
                    if not self.movej_p(place_pose, v=30):
                        self._enter_failure("PLACE_OBJECT", "移动到放置位姿失败", recover_state="MOVE_TO_DETECTION_POSE")
                        return
                    time.sleep(2)
                    print("[夹爪] 松开夹爪")
                    if not self.set_gripper_release():
                        self._enter_failure("PLACE_OBJECT", "放置阶段夹爪松开失败", recover_state="PLACE_OBJECT")
                        return
                    time.sleep(2)
                    self._mark_state_success("PLACE_OBJECT")
                    self.current_state = "RETURN_TO_DETECTION_POSE"
                else:
                    self._enter_failure(
                        "PLACE_OBJECT",
                        "获取当前位置失败，无法执行放置",
                        recover_state="MOVE_TO_DETECTION_POSE",
                    )
                    return

            elif self.current_state == "RETURN_TO_DETECTION_POSE":
                print("\n" + "="*60)
                print("状态: RETURN_TO_DETECTION_POSE - 返回检测位姿")
                print("="*60)
                print("[运动] 返回检测位姿")
                if not self.movej(self.detection_joint_angles, v=30):
                    self._enter_failure(
                        "RETURN_TO_DETECTION_POSE",
                        "抓取后返回检测位姿失败",
                        recover_state="MOVE_TO_DETECTION_POSE",
                    )
                    return
                time.sleep(2)
                print("\n" + "="*60)
                print("抓取验证单轮完成，机械臂已回到检测位姿")
                print("="*60)
                self._mark_state_success("RETURN_TO_DETECTION_POSE")
                self.current_state = "DONE"

            elif self.current_state == "RETURN_TO_HOME":
                print("\n" + "="*60)
                print("状态: RETURN_TO_HOME - 返回初始位姿")
                print("="*60)
                print("[运动] 返回初始位姿")
                if not self.movej(self.home_joint_angles, v=30):
                    self._enter_failure("RETURN_TO_HOME", "返回初始位姿失败", recover_state="RETURN_TO_HOME")
                    return
                time.sleep(2)
                print("\n" + "="*60)
                print("抓取流程完成！")
                print("="*60)
                self._mark_state_success("RETURN_TO_HOME")
                self.current_state = "FINISHED"

            elif self.current_state == "ERROR_RECOVERY":
                print("\n" + "="*60)
                print("状态: ERROR_RECOVERY - 执行恢复策略")
                print("="*60)
                print(f"[恢复] 最近错误: {self.last_error}")
                print("[恢复] 尝试先回到检测位姿")
                if self.movej(self.detection_joint_angles, v=30):
                    print(f"[恢复] 恢复成功，返回状态: {self.recovery_target_state}")
                    self.current_state = self.recovery_target_state
                else:
                    self.safe_stop_reason = f"恢复失败: {self.last_error}"
                    print("[恢复] 无法回到检测位姿，进入 SAFE_STOP")
                    self.current_state = "SAFE_STOP"

            elif self.current_state == "SAFE_STOP":
                print("\n" + "="*60)
                print("状态: SAFE_STOP - 安全停机")
                print("="*60)
                print(f"[安全停机] 原因: {self.safe_stop_reason}")
                print("[安全停机] 尝试释放夹爪，等待人工介入")
                self.set_gripper_release()
                self.current_state = "DONE"

            elif self.current_state == "FINISHED":
                print("\n流程结束")
                self.current_state = "DONE"

        except Exception as e:
            self._enter_failure(
                state_before_run,
                f"状态执行异常: {e}",
                recover_state="CHECK_OBJECT",
                max_retries=1,
            )

    def reset_state(self):
        """重置状态"""
        self.current_state = "INIT"
        self.retry_count = 0
        self.object_detected = False
        self.object_pose_in_base = None
        self.detection_attempts = 0
        self.state_failures = {}
        self.state_retry_limits = {
            "INIT": 2,
            "MOVE_TO_HOME": 2,
            "MOVE_TO_DETECTION_POSE": 2,
            "CHECK_OBJECT": 2,
            "MOVE_TO_PRE_GRASP": 2,
            "OPEN_GRIPPER": 2,
            "MOVE_TO_GRASP": 2,
            "CLOSE_GRIPPER": 2,
            "CHECK_GRASP_SUCCESS": 1,
            "PLACE_OBJECT": 2,
            "RETURN_TO_DETECTION_POSE": 2,
            "RETURN_TO_HOME": 2,
        }
        self.last_error = ""
        self.safe_stop_reason = ""
        self.recovery_target_state = "CHECK_OBJECT"
        print("[系统] 状态已重置")

    def wait_user_command_after_done(self):
        """
        进入 DONE 后等待用户输入命令。
        - 输入 R: 从初始状态重新运行
        - 输入 Q: 退出程序

        Returns:
            bool: True 表示继续运行，False 表示退出
        """
        print("\n" + "=" * 60)
        print("系统处于 DONE 状态，等待用户指令")
        print("输入 R 重新运行，输入 Q 退出程序")
        print("=" * 60)

        while True:
            cmd = input("[请输入指令 R/Q]: ").strip().upper()
            if cmd == "R":
                print("[用户指令] 收到 R，继续下一轮抓取验证")
                # 保持当前机械臂停在检测位姿，直接进入检测环节。
                self.current_state = "CHECK_OBJECT"
                self.retry_count = 0
                self.object_detected = False
                self.object_pose_in_base = None
                self.detection_attempts = 0
                self.state_failures = {}
                self.last_error = ""
                self.safe_stop_reason = ""
                return True
            if cmd == "Q":
                print("[用户指令] 收到 Q，程序即将退出")
                return False
            print("[用户指令] 无效输入，请输入 R 或 Q")
    
    def run_complete_cycle(self, cycles=1):
        """
        运行完整的抓取循环
        
        Args:
            cycles: 循环次数
        """
        print("\n" + "="*60)
        print(f"开始执行视觉抓取流程，计划运行 {cycles} 个循环")
        print("="*60)
        
        for cycle in range(cycles):
            print(f"\n{'='*60}")
            print(f"循环 {cycle+1}/{cycles}")
            print(f"{'='*60}")
            
            self.reset_state()
            
            # 运行状态机直到完成
            # while self.current_state != "FINISHED" and self.current_state != "DONE":
            while True:
                self.run_state_machine()
                if self.current_state == "DONE":
                    should_continue = self.wait_user_command_after_done()
                    if not should_continue:
                        print("\n[系统] 用户选择退出，结束运行")
                        return
                time.sleep(0.5)  # 短暂延时
            
            # if cycle < cycles - 1:
            #     print(f"\n完成第 {cycle+1} 个循环，准备开始下一个循环...")
            #     time.sleep(2)
        
        print("\n" + "="*60)
        print("所有抓取循环完成！")
        print("="*60)

    def shutdown(self):
        """释放 ROS2 资源"""
        if self.ros_node is not None:
            try:
                self.ros_node.destroy_node()
            except Exception:
                pass
            self.ros_node = None
        if ROS2_POSE_MSG_AVAILABLE and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

        
def main():
    """主函数"""
    print("\n" + "="*60)
    print("视觉抓取系统 - 最终版本")
    print("基于 requirements.md 完整实现")
    print("="*60)
    
    parser = argparse.ArgumentParser(description="RM 视觉抓取演示")
    parser.add_argument("--ip", default="192.168.1.18")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pose-topic", default="/pose_estimation_result")
    args = parser.parse_args()

    # 创建系统实例（仅支持真实机械臂 + DenseFusion topic）
    grasp_system = VisionGraspingSystem(
        ip=args.ip,
        port=args.port,
        pose_topic=args.pose_topic,
    )
    
    try:
        grasp_system.run_complete_cycle(cycles=1)
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        grasp_system.shutdown()
        print("\n" + "="*60)
        print("程序结束")
        print("="*60)


if __name__ == "__main__":
    main()
