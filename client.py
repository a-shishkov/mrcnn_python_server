import base64
import json
import socket
import skimage.io


def read(sock):
    data = sock.recv(4)
    if not data:
        return None

    read_len = int.from_bytes(data, "big")
    data = b""

    while len(data) < read_len:
        packet = sock.recv(read_len - len(data))
        if not packet:
            return None
        data += packet
    return data


def send(sock, data):
    sock.send(len(data).to_bytes(4, "big"))
    sock.sendall(data)


def client(ip, port, image_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")

        send(
            sock,
            json.dumps(
                {
                    "model": "damage",
                    "image": encoded,
                    "ananotations": None,
                    # "demo": True,
                }
            ).encode("utf-8"),
        )

        while True:
            response = read(sock)

            if response is None:
                return
            response = json.loads(response)
            print(response["response"])

            if response["response"] == "Message":
                print(response["message"])
            if response["response"] == "Results":
                return
