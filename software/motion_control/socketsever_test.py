import socket
import grasp_gift
# 设置服务器地址和端口
HOST = '127.0.0.1'  # 本地地址
PORT = 12345        # 监听端口

# 假设grasp_gift是您希望执行的函数

# 创建 TCP socket 服务器
def run_server():
    # 创建一个 TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # 绑定地址和端口
        server_socket.bind((HOST, PORT))

        # 开始监听客户端连接
        server_socket.listen()
        print(f"Server is listening on {HOST}:{PORT}...")

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
                        if message == 'red' or message == 'green':
                            conn.sendall(data)  # 发送数据给客户端

                            grasp_gift.main()

                        # 尝试将接收到的数据发送回客户端
                        try:
                            conn.sendall(data)  # 发送数据给客户端
                            print(f"Sent back: {message}")
                        except Exception as e:
                            print(f"Error sending data: {e}")
            except Exception as e:
                print(f"Error in connection: {e}")

# 启动服务器
if __name__ == "__main__":
    run_server()
