import time
from pyorbbecsdk import *
import numpy as np  # 导入NumPy库，用于数组和矩阵操作
import yaml
import json
import cv2
from software.vision.utils import frame_to_bgr_image, frame_to_rgb_frame

ESC_KEY = 27
MIN_DEPTH = 100  # 20mm
MAX_DEPTH = 500  # 10000mm


class TemporalFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            self.previous_frame = frame
            return frame
        result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


class OrbbecCamera:
    def __init__(self, device_id, config_extrinsic='./hand_eye_config.yaml'
                 ):
        # 获取设备列表
        self.context = Context()
        device_list = self.context.query_devices()
        self.device = device_list.get_device_by_index(device_id)
        self.sensor_list = self.device.get_sensor_list()
        self.device_info = self.device.get_device_info()
        self.config_path = config_extrinsic
        self.temporal_filter = TemporalFilter()
        self.config = Config()
        self.pipeline = Pipeline()
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        self.depth_profile = profile_list.get_default_video_stream_profile()
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self.color_profile = profile_list.get_default_video_stream_profile()

        self.extrinsic_matrix = []

        self.param = self.pipeline.get_camera_param()
        print(self.param)
        self.device.set_depth_work_mode("High Density")
        self.set_auto_exposure(True)
        self.device.set_bool_property(OBPropertyID.OB_PROP_DISPARITY_TO_DEPTH_BOOL, True)
        self.start_stream()

        self.depth_intrinsic = self.param.depth_intrinsic
        self.depth_fx = self.depth_intrinsic.fx  # 焦距 X
        self.depth_fy = self.depth_intrinsic.fy  # 焦距 Y
        self.depth_ppx = self.depth_intrinsic.cx  # 主点 X
        self.depth_ppy = self.depth_intrinsic.cy  # 主点 Y
        self.depth_distortion = self.param.depth_distortion  # 深度相机畸变系数

        self.rgb_intrinsic = self.param.rgb_intrinsic
        self.fx = self.rgb_intrinsic.fx  # RGB 相机焦距 X
        self.fy = self.rgb_intrinsic.fy  # RGB 相机焦距 Y
        self.ppx = self.rgb_intrinsic.cx  # RGB 相机主点 X
        self.ppy = self.rgb_intrinsic.cy  # RGB 相机主点 Y
        self.rgb_distortion = self.param.rgb_distortion  # RGB 相机畸变系数

    def get_device_name(self):
        return self.device_info.get_name()

    def get_device_pid(self):
        return self.device_info.get_pid()

    def get_serial_number(self):
        return self.device_info.get_serial_number()

    # 设置自动曝光模式
    def set_auto_exposure(self, auto_exposure: bool):
        # 设置自动曝光属性
        self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, auto_exposure)
        print(f"Auto exposure set to: {auto_exposure}")

    # 获取当前曝光值
    def get_current_exposure(self):
        return self.device.get_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT)

    # 直接设置曝光值
    def set_exposure(self, exposure_value: int):
        self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, exposure_value)
        print(f"Exposure set to: {exposure_value}")  # zhij#

    # 在当前曝光值的基础上调整
    def adjust_exposure(self, adjustment: int):
        # 获取当前曝光值
        curr_exposure = self.get_current_exposure()
        # 调整曝光
        new_exposure = curr_exposure + adjustment
        # 设置新的曝光值
        self.set_exposure(new_exposure)

    # 开关软件滤波
    def set_software_filter(self, soft_filter: bool):
        self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_SOFT_FILTER_BOOL, soft_filter)
        print(f"Software filter set to: {soft_filter}")

    # 重启设备
    def reboot(self):
        self.device.reboot()

    def start_stream(self, depth_stream=True, color_stream=True, enable_sync=True):
        '''开启色彩流'''
        if color_stream:
            self.config.enable_stream(self.color_profile)
        '''开启深度流'''
        if depth_stream:
            self.config.enable_stream(self.depth_profile)
            self.config.set_align_mode(OBAlignMode.SW_MODE)
        self.pipeline.start(self.config)

        if enable_sync:
            try:
                self.pipeline.enable_frame_sync()
            except Exception as e:
                print(e)
        self.param = self.pipeline.get_camera_param()
        print(self.param)
        self.depth_fx = self.param.depth_intrinsic.fx  # 焦距 X
        self.depth_fy = self.param.depth_intrinsic.fy  # 焦距 Y
        self.depth_ppx = self.param.depth_intrinsic.cx  # 主点 X
        self.depth_ppy = self.param.depth_intrinsic.cy  # 主点 Y
        self.depth_distortion = self.param.depth_distortion  # 深度相机畸变系数

        self.rgb_intrinsic = self.param.rgb_intrinsic
        self.rgb_fx = self.param.rgb_intrinsic.fx  # RGB 相机焦距 X
        self.rgb_fy = self.param.rgb_intrinsic.fy  # RGB 相机焦距 Y
        self.rgb_ppx = self.param.rgb_intrinsic.cx  # RGB 相机主点 X
        self.rgb_ppy = self.param.rgb_intrinsic.cy  # RGB 相机主点 Y
        self.rgb_distortion = self.param.rgb_distortion  # RGB 相机畸变系数
        print(self.param)

    def get_frames(self):
        '''先设置对齐方式，对齐后得到的depth_frame和color_frame的尺寸是一样的，可以直接根据像素点得到深度信息'''
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                # print("No frames")
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                # print("No depth frame")
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                # print("No color frame")
                continue
            point_clouds = frames.get_point_cloud(self.param)
            if point_clouds is None:
                print("No point clouds")
            break
        color_image = self.color_frame2color_image(color_frame)
        depth_image = self.depth_frame2depth_data(depth_frame, filter_on=True)

        return color_image, depth_image, depth_frame

    def get_depth_for_color_pixel(self, depth_frame, color_point):
        depth_data = self.depth_frame2depth_data(depth_frame, filter_on=True)
        center_x, center_y = color_point
        center_depth = 0
        if center_depth == 0:
            # 如果中心点的深度为0，则查找最近的不为0的深度值
            found = False
            search_radius = 1
            while not found:
                # 确定搜索范围的边界
                min_x = max(center_x - search_radius, 0)
                max_x = min(center_x + search_radius + 1, depth_data.shape[1])  # 使用 depth_frame 的宽度
                min_y = max(center_y - search_radius, 0)
                max_y = min(center_y + search_radius + 1, depth_data.shape[0])  # 使用 depth_frame 的高度

                # 遍历搜索范围内的每个像素点
                for x in range(min_x, max_x):
                    for y in range(min_y, max_y):
                        depth = depth_data[y, x]
                        if depth > 0:
                            center_depth = depth  # 获取最近的非零深度值
                            found = True
                            break
                    if found:
                        break

                if not found:
                    search_radius += 1  # 增加搜索半径
        return center_depth

    def color_frame2color_image(self, color_frame):
        color_image = frame_to_bgr_image(color_frame)
        return color_image

    def depth_frame2depth_data(self, depth_frame, filter_on=True):

        height = depth_frame.get_height()
        width = depth_frame.get_width()
        scale = depth_frame.get_depth_scale()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))
        depth_data = depth_data.astype(np.float32) * scale
        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
        depth_data = depth_data.astype(np.uint16)
        if filter_on:
            # Apply temporal filtering
            depth_data = self.temporal_filter.process(depth_data)
        return depth_data

    def show_depth_frame(self, depth_frame):
        depth_data = self.depth_frame2depth_data(depth_frame, filter_on=False)
        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        cv2.imshow("Depth Viewer", depth_image)

    def stop(self):
        self.pipeline.stop()

    def adjust_exposure_based_on_brightness(self, target_brightness=100):
        # 设置调整步长和曝光范围
        exposure_step = 50
        min_exposure = 30
        max_exposure = 10000

        while True:
            color_image, depth_image, depth_frame = self.get_frames()  # 获取一帧图像

            # 根据人眼感知计算亮度（加权平均法）
            print(color_image.shape)
            current_brightness = (
                    0.299 * color_image[:, :, 2] +  # Red 通道
                    0.587 * color_image[:, :, 1] +  # Green 通道
                    0.114 * color_image[:, :, 0]  # Blue 通道
            ).mean()

            print(f"当前亮度: {current_brightness}")
            # 根据亮度调整曝光值
            current_exposure = self.get_current_exposure()
            if current_brightness < target_brightness - 10:
                new_exposure = min(current_exposure + exposure_step, max_exposure)
                self.set_exposure(new_exposure)
                print(f"亮度过低，增加曝光值到: {new_exposure}")
            elif current_brightness > target_brightness + 10:
                new_exposure = max(current_exposure - exposure_step, min_exposure)
                self.set_exposure(new_exposure)
                print(f"亮度过高，减少曝光值到: {new_exposure}")
            else:
                print("亮度已调整至合理范围，无需进一步调整。")
                break

    def load_extrinsic(self):
        """从配置文件中加载外参."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                transformation_matrix = config['hand_eye_transformation_matrix']
                # 将外参矩阵转换为 NumPy 数组
                self.extrinsic_matrix = np.array(transformation_matrix)
        except Exception as e:
            raise RuntimeError(f"加载外参失败: {e}")


def initialize_connected_cameras():
    """初始化所有已连接的相机"""
    context = Context()  # 创建一个新的上下文对象
    device_list = context.query_devices()
    print(device_list)
    device_id_list = list(range(len(device_list)))
    devices = [OrbbecCamera(device_id) for device_id in device_id_list]  # 为每个设备ID创建一个Camera对象
    return devices  # 返回所有创建的Camera对象列表


def close_connected_cameras(cameras):
    """关闭所有已连接的相机"""
    for camera in cameras:  # 遍历所有Camera对象
        camera.stop()  # 关闭相机的数据流


def main():
    # 初始化连接的所有相机
    cameras = initialize_connected_cameras()
    temporal_filter = TemporalFilter(alpha=0.5)
    if len(cameras) == 0:
        # print("No cameras connected.")
        return

    # 假设只使用第一个相机
    camera = cameras[0]
    # camera.start_stream()
    try:
        # 获取一帧图像数据
        while True:
            color_image, depth_image, depth_frame = camera.get_frames()

            if color_image is None:
                print("failed to convert frame to image")
                continue
            cv2.imshow("Color Viewer", color_image)

            camera.show_depth_frame(depth_frame)
            cv2.waitKey(1)


    finally:
        # 关闭相机
        camera.stop()


if __name__ == "__main__":
    main()
