import socket
import time

from grasp_gift import RobotController
import threading

# 设置服务器地址和端口
HOST = '127.0.0.1'  # 本地地址
PORT = 6666  # 监听端口
controller = RobotController(conf_threshold=0.6)


# 创建 TCP socket 服务器
def run_server():
    # 创建一个 TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # 绑定地址和端口
        server_socket.bind((HOST, PORT))

        # 开始监听客户端连接
        server_socket.listen()
        print(f"Server is listening on {HOST}:{PORT}...")
        story_thread = None
        while True:
            try:
                # 接收客户端连接
                conn, addr = server_socket.accept()
                print(f"Connected by {addr}")

                # 使用连接处理数据
                with conn:
                    while True:
                        # 接收数据
                        data = conn.recv(1024)
                        if not data:
                            break  # 如果没有数据，退出当前连接

                        # 输出接收到的数据
                        message = data.decode('utf-8')
                        print(f"Received: {message}")

                        # 如果接收到的消息是 'red' 或 'green'，则调用 grasp_gift
                        if message == 'gift':
                            get_gift = False
                            while not get_gift:
                                time.sleep(0.1)
                                # 执行任务
                                controller.pre_execute()
                                controller.execute()

                                time.sleep(0.1)
                                # 发送执行完成信号
                                conn.sendall("ready".encode('utf-8'))
                                print("Sent: ready")

                                # 等待客户端回复 'done' 信号，确认可以继续执行 release_object
                                done_signal = conn.recv(1024).decode('utf-8')
                                if done_signal == 'yes':
                                    get_gift = True
                                    # 执行释放操作
                                    time.sleep(0.1)
                                    controller.release_object()
                                    conn.sendall("end".encode('utf-8'))
                                    print("Sent: end")
                                elif done_signal == 'no_gift':
                                    controller.release_object()
                                    get_gift = False
                        if message == 'circle':
                            # 执行任务
                            time.sleep(0.1)
                            controller.pre_circle()
                            conn.sendall("ready".encode('utf-8'))
                            print("Sent: ready")
                            controller.circle()
                            time.sleep(0.1)
                            # 发送执行完成信号
                            conn.sendall("end".encode('utf-8'))
                            print("Sent: end")
                        if message == 'greet':
                            # 执行任务
                            time.sleep(0.1)
                            controller.pre_greet()
                            conn.sendall("ready".encode('utf-8'))
                            print("Sent: ready")
                            controller.greet()
                            time.sleep(0.1)
                            # 发送执行完成信号
                            conn.sendall("end".encode('utf-8'))
                            print("Sent: end")

                        if message == 'story':
                            controller.pre_story()
                            time.sleep(0.1)
                            # 执行任务
                            conn.sendall("start".encode('utf-8'))
                            print("Sent: start")

                            story_thread = threading.Thread(target=controller.story)
                            story_thread.start()
                            time.sleep(0.1)
                            # 发送执行完成信号
                        if message == 'story_end':
                            # 执行任务
                            if story_thread and story_thread.is_alive():
                                controller.stop_story()  # 通知线程停止
                                story_thread.join()  # 等待线程结束
                            else:
                                pass
                        if message == 'talk':
                            time.sleep(0.1)
                            # 执行任务
                            story_thread = threading.Thread(target=controller.story)
                            story_thread.start()
                            # 发送执行完成信号
                        if message == 'talk_end' or message == 'talktalk_end':
                            # 执行任务
                            if story_thread and story_thread.is_alive():
                                controller.stop_story()  # 通知线程停止
                                story_thread.join()  # 等待线程结束
                            else:
                                pass
            except Exception as e:
                print(f"Error in connection: {e}")


# 启动服务器
if __name__ == "__main__":
    run_server()
