import yaml
import time

from pyorbbecsdk import Context

from hans_robot.CPS import CPSClient  # 假设有一个机器人控制的库
from software.vision.orbbec_camera import OrbbecCamera  # Realsense 相机类
from jodell_gripper.RG.rg_api import ClawOperation  # 手爪控制类
from ultralytics import YOLO  # YOLO 检测库
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import random  # 导入random模块
import threading


class RobotController:
    def __init__(self, box_id=0, rbt_id=0, position_file='../config/position.yaml',
                 model_path='../config/bottle_head_results/weights/best.pt',
                 conf_threshold=0.8,
                 calibration_file='../config/config.yaml'):
        # 初始化参数
        self.box_id = box_id
        self.rbt_id = rbt_id
        self.position_file = position_file
        self.calibration_file = calibration_file
        self.client = CPSClient()

        # 连接到电箱和控制器
        self.client.HRIF_Connect(self.box_id, '192.168.11.7', 10003)
        self.client.HRIF_Connect2Controller(self.box_id)

        # 获取当前位置
        self.current_pose = self.client.getTCPPose()  # 读取当前位置
        print(f"Initial Current Pose: {self.current_pose}")

        # 读取目标位置
        self.positions = self.read_position()
        self.joint = self.read_position("joint")

        """初始化所有已连接的相机"""
        context = Context()  # 创建一个新的上下文对象
        device_list = context.query_devices()
        print(device_list)
        device_id_list = list(range(len(device_list)))
        # 初始化 Realsense 相机
        devices = [OrbbecCamera(device_id) for device_id in device_id_list]  # 为每个设备ID创建一个Camera对象

        self.camera = devices[0]
        # 读取标定信息
        self.calibration_data = self.read_calibration()
        # 初始化手爪控制
        self.gripper = ClawOperation(com_port="/dev/ttyUSB0", baud_rate=115200)  # 假设串口设备在 /dev/ttyUSB0
        self.gripper.connect()

        # 初始化 YOLO 模型
        self.model = YOLO(model_path, task="val")  # 确保使用正确的模型路径和设备

        # 设置置信度阈值
        self.conf_threshold = conf_threshold
        # self.gripper.enable_claw(claw_id=9, enable=True)

        self.gripper.open()
        self._stop_event = threading.Event()

    def read_calibration(self, tag="hand_eye_transformation_matrix"):
        """读取标定信息"""
        with open(self.calibration_file, 'r') as file:
            calibration = yaml.safe_load(file)
        return calibration[tag]

    def read_position(self, tag='positions'):
        """读取position.yaml文件，获取目标位置"""
        with open(self.position_file, 'r') as file:
            positions = yaml.safe_load(file)
        return positions[tag]

    def calculate_position(self, detection, depth_frame, center_depth=333):
        # 计算检测到的目标物体的3D位置的函数。

        # 假设检测到的第一个目标为所需目标，获取其四个顶点的坐标
        # 展开detection并获取四个角的坐标
        x1, y1, x2, y2 = detection.flatten()

        # 计算目标的中心点
        center_x = int((x1 + x2) // 2)
        center_y = int((y1 + y2) // 2)
        # 获取中心点的深度值
        center_depth = self.camera.get_depth_for_color_pixel(depth_frame=depth_frame,
                                                             color_point=[center_x, center_y])
        print(center_depth)
        center_depth = center_depth
        # 计算实际深度（Z轴）
        real_z = center_depth / 1000

        fx, fy = self.camera.rgb_fx, self.camera.rgb_fy  # 获取相机的焦距参数
        real_x = (center_x - self.camera.ppx) * real_z / fx  # 计算目标的实际X坐标
        real_y = (center_y - self.camera.ppy) * real_z / fy  # 计算目标的实际Y坐标

        print(
            f"目标位置（使用depth_image计算）（相机坐标系）：x={real_x}, y={real_y}, z={real_z}°")
        return real_x, real_y, real_z, 0  # 返回目标的实际坐标和旋转角度

    def detect_object(self, color_image, show=False):
        """
        使用YOLO进行物体检测并返回物体的中心点。

        参数:
        show (bool): 如果为True，将显示检测结果图像。

        返回:
        object_center (tuple): 物体的中心点坐标 (x, y)
        """
        # 获取相机帧

        # 使用 YOLO 进行物体检测
        results = self.model.predict(color_image)

        # 显示结果
        if show:
            for item in results:
                item.show(font_size=1, line_width=2)

        # 获取结果中的边界框、标签和置信度
        boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取所有边界框
        confidences = results[0].boxes.conf.cpu().numpy()  # 获取每个框的置信度
        labels = results[0].names  # 获取物体标签

        # 记录最大置信度的 'bottle_head' 边界框
        max_confidence = -1
        max_confidence_box = None

        # 遍历所有边界框，找到 'bottle_head' 并筛选出置信度最高的
        for idx, label in enumerate(results[0].boxes.cls.cpu().numpy()):
            if labels.get(int(label)) == 'bottle_head' and confidences[idx] >= 0.9:
                if confidences[idx] > max_confidence:
                    max_confidence = confidences[idx]
                    max_confidence_box = boxes[idx]

        # 输出置信度最高的 'bottle_head' 边界框（xyxy）
        if max_confidence_box is not None:
            return max_confidence_box
        else:
            print("没有找到置信度高于0.9的 'bottle_head'。")
            return None

    def perform_grab(self, object_center):
        """执行抓取动作"""

        def euler_to_transformation_matrix(pose):
            pose = np.array(pose)
            pose[0:3] = pose[0:3] / 1000
            x, y, z, roll, pitch, yaw = pose
            """
            将位姿 [x, y, z, roll, pitch, yaw] 转换为 4x4 齐次变换矩阵。

            参数:
            x, y, z (float): 物体的位置坐标
            roll, pitch, yaw (float): 欧拉角，绕 X、Y、Z 轴的旋转角度，单位为度

            返回:
            numpy array: 4x4 齐次变换矩阵
            """
            # 创建旋转矩阵（欧拉角转换为旋转矩阵）
            r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
            rotation_matrix = r.as_matrix()

            # 创建 4x4 齐次变换矩阵
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix  # 将旋转矩阵填入左上角
            transformation_matrix[:3, 3] = [x, y, z]  # 将平移向量填入最后一列

            return transformation_matrix

        def create_transformation_matrix(object_center, theta):
            """
            计算物体的 4x4 齐次变换矩阵。

            参数:
            object_center (numpy array): 物体的平移坐标 [x, y, z]
            theta (float): 绕 Z 轴的旋转角度，单位为度

            返回:
            numpy array: 4x4 齐次变换矩阵
            """
            # 将角度转换为弧度，并使用 scipy 创建旋转矩阵（绕 Z 轴的旋转）
            r = R.from_euler('z', theta, degrees=True)
            R_z = r.as_matrix()  # 获取旋转矩阵

            # 构造4x4的齐次变换矩阵
            T = np.eye(4)  # 4x4单位矩阵
            T[:3, :3] = R_z  # 将旋转矩阵填入左上角
            T[:3, 3] = object_center  # 将平移向量填入最后一列

            return T

        def transformation_matrix_to_xyzrpy(matrix):
            # 提取平移向量 (x, y, z)
            translation = matrix[:3, 3] * 1000

            # 提取旋转矩阵 (3x3)
            rotation_matrix = matrix[:3, :3]

            # 计算旋转的欧拉角（绕 Z, Y, X 轴的旋转）
            r = R.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)

            # 返回平移和旋转欧拉角
            return np.concatenate([translation, euler_angles])

        def create_end_to_finger_matrix(x=0.0, y=0.0, z=0.18, theta=0):
            """
            创建一个 4x4 齐次变换矩阵。

            参数:
            x, y, z (float): 物体的平移坐标
            theta (float): 绕 Z 轴的旋转角度，单位为度

            返回:
            numpy array: 4x4 齐次变换矩阵
            """
            # 创建单位旋转矩阵（没有旋转）
            R = np.eye(3)

            # 构造4x4的齐次变换矩阵
            T = np.eye(4)
            T[:3, :3] = R  # 将旋转矩阵填入左上角
            T[:3, 3] = [x, y, z]  # 将平移向量填入最后一列

            return T

        theta = float(object_center[-1])
        object_center = np.array(object_center[0:3])
        T_camera_to_target = create_transformation_matrix(object_center, theta)
        T_base_to_end = euler_to_transformation_matrix(self.client.getTCPPose())
        T_end_to_camera = np.array(self.read_calibration())
        base_to_target = T_base_to_end @ T_end_to_camera @ T_camera_to_target
        T_end_to_finger = create_end_to_finger_matrix()
        aim_base_to_end = base_to_target @ np.linalg.inv(T_end_to_finger)
        aim_pose = transformation_matrix_to_xyzrpy(aim_base_to_end)
        aim_pose[3:6] = 119, 17, -166
        pre_grasp_pose = aim_pose.copy()  # 抓取前上方100mm
        grasp_pose = aim_pose.copy()  # 抓取时的高度
        after_grasp_pose = aim_pose.copy()  # 抓取时的高度
        pre_grasp_pose[1] = pre_grasp_pose[1]-100
        grasp_pose[1] = grasp_pose[1]+20  # 抓取时的高度
        grasp_pose[2] = grasp_pose[2]-20  # 抓取时的高度

        after_grasp_pose[2] = after_grasp_pose[2]+100

        # 移动到物体上方100mm
        print("Moving above the object...")
        self.client.move_robot(pre_grasp_pose)

        # 向下移动到抓取高度
        print("Moving down to grab the object...")
        self.client.move_robot(grasp_pose)
        self.gripper.close()
        # 控制手爪抓取物体
        print("Closing gripper to grab the object...")
        # self.gripper.set_force(50)  # 设置夹持力
        time.sleep(1)

        # 抓取完成后升高100mm
        print("Raising after grab...")
        ret = self.client.move_robot(after_grasp_pose)
        while ret != 0:
            after_grasp_pose[2] = after_grasp_pose[2] - 20
            ret = self.client.move_robot(after_grasp_pose)

    def release_object(self):
        """释放物体"""
        print("Opening gripper to release the object...")
        self.gripper.open()

    def pre_execute(self):
        # if self.current_pose[2] < 450:
        #     print(f"Current height {self.current_pose[2]} is less than 500. Moving to 500...")
        #     self.client.move_robot(target_pose=[self.current_pose[0], self.current_pose[1], 450, 0, 0, 0])
        init_joint = self.joint['init_joint']
        self.client.moveByJoint(target_joint=init_joint)
        # self.camera.adjust_exposure_based_on_brightness(target_brightness=158)

    def execute(self):
        """执行完整的任务流程"""
        # 1. 检查机器人当前高度并调整到500mm以上

        # 2. 捕获物体位置
        while True:
            time.sleep(0.1)
            color_image, depth_image, depth_frame = self.camera.get_frames()
            print("Captured frames from RealSense.")
            detection = self.detect_object(color_image, show=True)
            if detection is None or detection is [] or detection.shape[0] == 0:
                print("Object detection failed. Exiting.")
                return
            break
        object_center = self.calculate_position(detection, depth_frame)

        # 3. 执行抓取动作
        self.perform_grab(object_center)

        # 4. 移动到目标位置并释放物体
        gift_joint_release = self.joint['release_joint']
        self.client.moveByJoint(gift_joint_release)

    def draw_circle(self, boxID, rbtID, center, radius, y_height, num_points=36):
        """
        Draws a circle in 3D space by moving the robot along the calculated points of the circle.

        :param boxID: The ID of the box (robot system)
        :param rbtID: The ID of the robot
        :param center: The center of the circle (x, y, z)
        :param radius: The radius of the circle
        :param z_height: The constant Z height to keep the robot's end-effector at during the circle
        :param num_points: The number of points to calculate for the circle's circumference (default 36)
        """
        # Initialize the robot object

        # Generate circle points
        poses = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points  # Divide full circle into segments
            x = center[0] + radius * math.cos(angle)
            z = center[2] + radius * math.sin(angle)
            poses.append([x, y_height, z, -90.0, -20.0, -0.00])  # z_height is constant

        # Execute movement command to each pose
        dServoTime = 1
        dLookaheadTime = 0

        self.client.HRIF_StartServo(boxID, rbtID, dServoTime, dLookaheadTime)
        print(poses)

        for pose in poses:
            ucs = [0, 0, 0, 0, 0, 0]  # Placeholder UCS (could be set based on robot's configuration)
            tcp = [0, 0, 0, 0, 0, 0]  # Placeholder TCP (could be set based on the tool's setup)
            res = self.client.HRIF_PushServoP(boxID, rbtID, pose, ucs, tcp)  # Call to HRIF_PushServoP
            # print(res)
            time.sleep(dServoTime)  # Small delay between movements to avoid command overlap

    def pre_circle(self):
        cycle_joint_init = self.joint["cycle_joint_init"]
        self.client.moveByJoint(cycle_joint_init)

    def circle(self):
        # Setup parameters for the circle drawing
        boxID = 0  # Replace with the actual box ID
        rbtID = 0  # Replace with the actual robot ID
        center = [0, 400, 800]  # Example center point for the circle (x, y, z)
        radius = 50  # Radius of the circle
        y_height = 400  # Height at which the robot end-effector will move along the circle

        # Call draw_circle function to make the robot draw the circle
        self.gripper.run_with_param(claw_id=9, force=255, speed=255, position=5)
        self.draw_circle(boxID, rbtID, center, radius, y_height, num_points=36)

    def pre_greet(self):
        greet_joint_init = self.joint["greet_joint_init"]
        self.client.moveByJoint(greet_joint_init)

    def greet(self):
        # greet_point = self.positions["greet"]
        # self.client.move_robot(greet_point)
        greet_joint_init = self.joint["greet_joint_init"]
        greet_joint_left = self.joint["greet_joint_left"]
        greet_joint_right = self.joint["greet_joint_right"]
        self.client.moveByJoint(greet_joint_left)
        self.client.moveByJoint(greet_joint_right)
        self.client.moveByJoint(greet_joint_init)

        # self.client.move_robot(greet_point)

    def pre_story(self):
        greet_point = self.joint["greet_joint_init"]
        self.client.moveByJoint(greet_point)

    def story(self):
        while not self._stop_event.is_set():
            random_position = random.randint(0, 255)  # 生成100到1000之间的随机位置
            self.gripper.run_with_param(claw_id=9, force=255, speed=255, position=random_position)  # 设置随机位置并等待到达
        self.gripper.run_with_param(claw_id=9, force=255, speed=255, position=255)  # 设置随机位置并等待到达
        self._stop_event.clear()

    def stop_story(self):
        print("停止finger运动")
        self._stop_event.set()  # 设置事件，通知线程停止


def main():
    controller = RobotController(conf_threshold=0.6)
    controller.pre_execute()
    # 执行任务
    controller.execute()
    # time.sleep(1)
    # controller.release_object()


def story_():
    controller = RobotController(conf_threshold=0.6)
    controller.pre_story()
    controller.story()


def circle_():
    controller = RobotController(conf_threshold=0.6)
    controller.circle()


def greet_():
    controller = RobotController(conf_threshold=0.6)
    controller.greet()


# 使用示例
if __name__ == "__main__":
    main()
    # greet_()
