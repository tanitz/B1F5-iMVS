import socket


server_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket1.bind(('127.0.0.1', 5001))

server_socket1.close()
