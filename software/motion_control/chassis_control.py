import rospy
import os
from std_msgs.msg import String

class SimpleActionPublisher:
    def __init__(self, topic_name='/simple_action/command', master_uri="http://172.16.3.108:11311", hostname="172.16.3.209"):
        # 设置 ROS 环境变量
        os.environ['ROS_MASTER_URI'] = master_uri
        os.environ['ROS_HOSTNAME'] = hostname

        # 初始化 ROS 节点
        rospy.init_node('simple_action_talker', anonymous=True)

        # 创建发布者
        self.pub = rospy.Publisher(topic_name, String, queue_size=10)

        # 创建订阅者来监听状态消息
        self.state_subscriber = rospy.Subscriber("/simple_action/state", String, self.state_callback)

        # 保存返回的 ID 和状态
        self.response_id = None
        self.status_received = False

        rospy.loginfo("SimpleActionPublisher initialized and subscriber created.")

    def publish_message(self, message):
        # 发布消息
        rospy.loginfo(f"Publishing message: {message}")
        self.pub.publish(message)

    def state_callback(self, msg):
        # 解析返回的状态消息
        rospy.loginfo(f"Received state message: {msg.data}")  # 打印接收到的消息
        state_message = msg.data.split(',')

        rospy.loginfo(f"Parsed state message: {state_message}")  # 调试输出，查看解析后的消息

        if len(state_message) >= 2:
            action = state_message[0]
            status = state_message[1]

            if status == "SUCCESS" and len(state_message) > 2:
                # 如果状态是 SUCCESS，获取并保存 ID
                self.response_id = state_message[2]
                rospy.loginfo(f"Success! ID: {self.response_id}")
                self.status_received = True  # 标记收到成功状态
            elif status == "FAIL":
                # 如果状态是 FAIL，抛出错误并标记状态已收到
                error_msg = "Action failed: " + ",".join(state_message[2:])
                rospy.logerr(error_msg)
                self.status_received = True  # 标记收到失败状态
                raise Exception(error_msg)
            elif status == "RUNNING":
                rospy.loginfo("Action is still running...")
            else:
                rospy.loginfo(f"Unknown status: {status}")

    def start(self, message):
        # 等待直到节点初始化
        rospy.sleep(1)
        # 调用发布消息的方法
        self.publish_message(message)

        # 持续等待直到收到状态消息（SUCCESS 或 FAIL）
        rospy.loginfo("Waiting for action status...")

        # 循环等待，直到收到状态消息
        while not self.status_received:
            rospy.sleep(0.5)  # 每次循环等待 0.5 秒

        # 如果收到成功的消息，返回 ID
        if self.response_id:
            return self.response_id
        else:
            return "No successful response received"


if __name__ == '__main__':
    try:
        # 创建 SimpleActionPublisher 实例并发布消息
        publisher = SimpleActionPublisher()  # 使用默认参数
        message = "MOVE_STRAIGHT,bee638cc-af08-41ce-8295-c8b8584e5473"
        result = publisher.start(message)

        # 输出返回的 ID 或错误信息
        rospy.loginfo(f"Result: {result}")

        # 启动 ROS 事件循环
        # rospy.spin()

    except rospy.ROSInterruptException as e:
        rospy.logerr(f"Error: {e}")
