import cv2
import os

from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile
from utils import frame_to_bgr_image

ESC_KEY = 27


def get_unique_filename(save_dir, base_filename):
    """
    检查文件夹中是否已经存在该文件名，若存在，则递增数字，直到生成一个新的唯一文件名。
    """
    filename = os.path.join(save_dir, base_filename)
    count = 1
    # 如果文件已经存在，递增文件名
    while os.path.exists(filename):
        filename = os.path.join(save_dir, f"{base_filename[:-4]}_{count}.png")
        count += 1
    return filename


def main():
    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    pipeline.start(config)

    # 创建一个目录来保存图片
    save_dir = "captured_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            cv2.imshow("Color Viewer", color_image)
            key = cv2.waitKey(1)

            # 按下 'r' 键保存图片，并确保文件名唯一
            if key == ord('r'):
                base_filename = "image.png"
                unique_filename = get_unique_filename(save_dir, base_filename)
                cv2.imwrite(unique_filename, color_image)
                print(f"Saved image as {unique_filename}")

            # 按下 'q' 或 ESC 退出
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break
    pipeline.stop()


if __name__ == "__main__":
    main()
