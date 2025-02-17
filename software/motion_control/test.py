import rospy


def talker():
    rospy.init_node('talker', anonymous=True)
    rospy.loginfo("Hello, ROS!")
    rospy.spin()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
