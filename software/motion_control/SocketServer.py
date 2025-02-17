import socket
from grasp_elibot import RobotController


class SocketServer:
    def __init__(self, host='192.168.1.54', port=6666):
        self.host = host
        self.port = port
        self.client_socket = None  # 成员数据
        self.client_address = None
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.robot = RobotController(conf_threshold=0.3)
        print(f"Server started on {self.host}:{self.port}")

    def start(self):
        print("Waiting for connection...")
        while True:
            self.client_socket, self.client_address = self.server_socket.accept()
            print(f"Connection established with {self.client_address}")
            self.handle_client(self.client_socket)

    def handle_client(self, client_socket):
        try:
            while True:
                data = self.receive_data(client_socket)
                if not data:
                    break

                print(f"Received: {data}")
                if data == b"grasp":
                    self.grasp()
                else:
                    self.send_data(client_socket, b"Unknown command")
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
            print("Connection closed")

    def receive_data(self, client_socket):
        try:
            data = client_socket.recv(1024)
            return data
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None

    def send_data(self, client_socket, data):
        try:
            client_socket.sendall(data)
        except Exception as e:
            print(f"Error sending data: {e}")

    def grasp(self):
        self.send_data(self.client_socket, b"grasp")
        while True:
            data = self.receive_data(self.client_socket)
            if not data:
                print(f"Received: {data}")
                break
        self.robot.pre_execute()
        self.robot.execute(data)
        self.send_data(self.client_socket, data)

    print("Connection closed")


if __name__ == "__main__":
    server = SocketServer()
    server.start()
