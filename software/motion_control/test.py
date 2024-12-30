from scipy.spatial.transform import Rotation as R
import numpy as np

# 初始 RPY (roll, pitch, yaw)
rpy_initial = [180.0, 0.0, 20.0]

# 将 RPY 转换为旋转矩阵
rotation_initial = R.from_euler('xyz', rpy_initial, degrees=True)

# 绕 X 轴旋转 90 度
rotation_x_90 = R.from_euler('x', 90, degrees=True)

# 组合旋转
combined_rotation = rotation_x_90 * rotation_initial

# 获取组合后的旋转矩阵
rotation_matrix = combined_rotation.as_matrix()

# 获取组合后的欧拉角
combined_rpy = combined_rotation.as_euler('xyz', degrees=True)

print("Combined Rotation Matrix:")
print(rotation_matrix)
print("\nCombined RPY:")
print(combined_rpy)
