import socket
import time
import re

HOST, PORT = "192.168.2.112", 10100
sock = socket.socket()
#sock.connect((HOST, PORT))


def connect():
    # StartDummy
    data = "attach 1"
    sock.sendall(bytes(data + "\n", "utf-8"))
    received = str(sock.recv(1024), "utf-8")
    # EndDummy

    print("Enabling Power")
    data = "hp 1"
    sock.sendall(bytes(data + "\n", "utf-8"))
    received = str(sock.recv(1024), "utf-8")
    print("Received:{}".format(received))
    chr0 = received[0]
    if chr0 != '0':
        print("Error enabling Power ")
        exit()
    print("Wait 10s")
    time.sleep(10.0)

    print("Attaching Bot")
    data = "attach 1"
    sock.sendall(bytes(data + "\n", "utf-8"))
    received = str(sock.recv(1024), "utf-8")
    chr0 = received[0]
    if chr0 != '0':
        print("Error attaching Bot ")
        exit()

    print("Homing Bot")
    data = "home"
    sock.sendall(bytes(data + "\n", "utf-8"))
    received = str(sock.recv(1024), "utf-8")
    chr0 = received[0]
    if chr0 != '0':
        print("Error homing Bot ")
        exit()

    print("Connected")


def where():
    data = "0 481.2 5.179 -16.538 0 180 180"
    num = re.findall(r"[-+]?(?:\d*\.*\d+)", data)
    x = num[1]
    y = num[2]
    z = num[3]
    #print("x:", x)
    #print("y:", y)
    #print("z:", z)
    return x,y,z


def wherexyz():
    # Recived:0 481.2 5.179 -16.538 0 180 180
    data = "wherec"
    sock.sendall(bytes(data + "\n", "utf-8"))
    received = str(sock.recv(1024), "utf-8")
    chr0 = received[0]
    if chr0 != '0':
        sock.sendall(bytes(data + "\n", "utf-8"))
        received = str(sock.recv(1024), "utf-8")
    num = re.findall(r"[-+]?(?:\d*\.*\d+)", received)
    x = num[1]
    y = num[2]
    z = num[3]
    print("Sent:{}".format(data))
    print("Received:{}".format(received))
    # print("x:", x)
    # print("y:", y)
    # print("z:", z)
    return x, y, z


def testConnection():
    connect()
    x, y, z = wherexyz()
    print("x:", x)
    print("y:", y)
    print("z:", z)
    enableDIO(33)
    movexyz(400.0, 15.0, -5.0)
    movexyz(350.0, 15.0, -5.0)
    movexyz(400.0, 15.0, -5.0)
    disableDIO(33)
    disconnect()

def movexyz(x=495.0, y=10.0, z=-2.0):
    print("Moving to: ")
    print("x: " + str(x))
    print("y: " + str(y))
    print("z: " + str(z))
    data = "movec " + str(x) + " " + str(y) + " " + str(z) + " 0 180 180"
    print(data)
    sock.sendall(bytes(data + "\n", "utf-8"))
    received = str(sock.recv(1024), "utf-8")
    chr0 = received[0]
    if chr0 != '0':
        sock.sendall(bytes(data + "\n", "utf-8"))
        received = str(sock.recv(1024), "utf-8")
    print("Sent:{}".format(data))
    print("Received:{}".format(received))


def enableDIO(Port):
    data = "sig " + str(Port) + " 1"
    print(data)
    sock.sendall(bytes(data + "\n", "utf-8"))
    received = str(sock.recv(1024), "utf-8")
    chr0 = received[0]
    if chr0 != '0':
        sock.sendall(bytes(data + "\n", "utf-8"))
        received = str(sock.recv(1024), "utf-8")
    print("Sent:{}".format(data))
    print("Received:{}".format(received))


def disableDIO(Port):
    data = "sig " + str(Port) + " 0"
    print(data)
    sock.sendall(bytes(data + "\n", "utf-8"))
    received = str(sock.recv(1024), "utf-8")
    chr0 = received[0]
    if chr0 != '0':
        sock.sendall(bytes(data + "\n", "utf-8"))
        received = str(sock.recv(1024), "utf-8")
    print("Sent:{}".format(data))
    print("Received:{}".format(received))


def disconnect():
    movexyz(x=495.0, y=10.0, z=-2.0)
    print("Going home")
    print("Wait 15s")
    time.sleep(15.0)
    print("Disabling Power")
    data = "hp 0"
    sock.sendall(bytes(data + "\n", "utf-8"))
    received = str(sock.recv(1024), "utf-8")
    print("Received:{}".format(received))
    chr0 = received[0]
    if chr0 != '0':
        print("Error enabling Power ")
        exit()

    sock.detach()
