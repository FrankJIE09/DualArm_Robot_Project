import math
import time
from hans_robot.CPS import CPSClient  # Replace with the actual class name containing HRIF_PushServoP


def draw_circle(boxID, rbtID, center, radius, z_height, num_points=36):
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
        y = center[1] + radius * math.sin(angle)
        poses.append([x, y, z_height])  # z_height is constant

    # Execute movement command to each pose
    for pose in poses:
        ucs = [0, 0, 0]  # Placeholder UCS (could be set based on robot's configuration)
        tcp = [0, 0, 0]  # Placeholder TCP (could be set based on the tool's setup)
        robot.HRIF_PushServoP(boxID, rbtID, pose, ucs, tcp)  # Call to HRIF_PushServoP
        time.sleep(0.1)  # Small delay between movements to avoid command overlap


def main():
    robot = CPSClient()  # Replace with the correct class name and initialization if needed

    # Setup parameters for the circle drawing
    boxID = 1  # Replace with the actual box ID
    rbtID = 1  # Replace with the actual robot ID
    center = [0, 675, 700]  # Example center point for the circle (x, y, z)
    radius = 10  # Radius of the circle
    z_height = 10  # Height at which the robot end-effector will move along the circle

    # Call draw_circle function to make the robot draw the circle
    draw_circle(boxID, rbtID, center, radius, z_height, num_points=36)


if __name__ == "__main__":
    main()
