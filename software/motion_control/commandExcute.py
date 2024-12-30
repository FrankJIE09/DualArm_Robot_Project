# 定义不同功能的函数

import socket
import time


# 定义不同功能的函数

def handle_red():
    try:
        # 创建一个 TCP socket 客户端
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            # 尝试连接到 127.0.0.1 和端口 12345
            client_socket.connect(('127.0.0.1', 7787))

            # 发送字符串 '1'
            client_socket.sendall(b'story')

            # 如果需要接收返回消息，可以调用 recv()
            response = client_socket.recv(1024)

            if response:
                print("Received:", response.decode('utf-8'))
            else:
                print("No response received from the server.")
            time.sleep(2)
            client_socket.sendall(b'story_end')


        return "Executed red-related operation and sent '1' to 127.0.0.1"

    except ConnectionRefusedError:
        # 如果连接失败，捕获异常并输出连接失败
        return "连接失败"


def handle_green():
    try:
        # 创建一个 TCP socket 客户端
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            # 尝试连接到 127.0.0.1 和端口 12345
            client_socket.connect(('127.0.0.1', 12345))

            # 发送字符串 '1'
            client_socket.sendall(b'green')

            # 如果需要接收返回消息，可以调用 recv()
            response = client_socket.recv(1024)

            if response:
                print("Received:", response.decode('utf-8'))
            else:
                print("No response received from the server.")

        return "Executed red-related operation and sent '1' to 127.0.0.1"

    except ConnectionRefusedError:
        # 如果连接失败，捕获异常并输出连接失败
        return "连接失败"


def tell_story():
    try:
        # 创建一个 TCP socket 客户端
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            # 尝试连接到 127.0.0.1 和端口 12345
            client_socket.connect(('127.0.0.1', 12345))

            # 发送字符串 '1'
            client_socket.sendall(b'story')

            # 如果需要接收返回消息，可以调用 recv()
            response = client_socket.recv(1024)

            if response:
                print("Received:", response.decode('utf-8'))
            else:
                print("No response received from the server.")

        return "Executed red-related operation and sent '1' to 127.0.0.1"

    except ConnectionRefusedError:
        # 如果连接失败，捕获异常并输出连接失败
        return "连接失败"


def greet():
    try:
        # 创建一个 TCP socket 客户端
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            # 尝试连接到 127.0.0.1 和端口 12345
            client_socket.connect(('127.0.0.1', 12345))

            # 发送字符串 '1'
            client_socket.sendall(b'greet')

            # 如果需要接收返回消息，可以调用 recv()
            response = client_socket.recv(1024)

            if response:
                print("Received:", response.decode('utf-8'))
            else:
                print("No response received from the server.")

        return "Executed red-related operation and sent '1' to 127.0.0.1"

    except ConnectionRefusedError:
        # 如果连接失败，捕获异常并输出连接失败
        return "连接失败"


def draw_circle():
    try:
        # 创建一个 TCP socket 客户端
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            # 尝试连接到 127.0.0.1 和端口 12345
            client_socket.connect(('127.0.0.1', 12345))

            # 发送字符串 '1'
            client_socket.sendall(b'circle')

            # 如果需要接收返回消息，可以调用 recv()
            response = client_socket.recv(1024)

            if response:
                print("Received:", response.decode('utf-8'))
            else:
                print("No response received from the server.")

        return "Executed red-related operation and sent '1' to 127.0.0.1"

    except ConnectionRefusedError:
        # 如果连接失败，捕获异常并输出连接失败
        return "连接失败"


# 主函数：根据接收到的字符串的第一部分选择相应的操作
def execute_command(command: str) -> str:
    # 将命令按空格分成两部分
    command_parts = command.split(",", 1)  # 只分割一次

    # 定义指令映射字典
    command_map = {
        "红色的礼物": handle_red,
        # "绿色的礼物": handle_green,
        # "故事": tell_story,
        # "打招呼": greet,
        # "画圈": draw_circle
    }

    # 获取命令的第一部分
    command_key = command_parts[0]

    # 查找并执行相应的函数，如果没有找到则返回默认消息
    return command_map.get(command_key, lambda: "Unrecognized command.")()


# 测试用例（如果直接运行此脚本，会调用下面的部分）
if __name__ == "__main__":
    # 示例测试
    test_commands = [
        "红色的礼物,1",
        # "绿色的礼物,1",
        # "故事,1",
        "打招呼,1",
        # "画圈,1",
        # "其他指令"
    ]

    for command in test_commands:
        print(f"Input command: {command} -> Output: {execute_command(command)}")
