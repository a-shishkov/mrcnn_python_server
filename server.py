import base64
import json
import os
import socket
import threading
import socketserver
import mrcnn_inference
import skimage.io


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def read(self):
        data = self.request.recv(4)
        if not data:
            return None

        read_len = int.from_bytes(data, "big")
        data = b""

        while len(data) < read_len:
            packet = self.request.recv(read_len - len(data))
            if not packet:
                return None
            data += packet
        return data

    def send(self, data):
        self.request.send(len(data).to_bytes(4, "big"))
        self.request.sendall(data)

    def handle(self):
        while True:
            data = self.read()

            if data is None:
                return

            data = json.loads(data)

            model = data["model"]
            print(self.client_address)

            input_folder = os.path.join(
                "results",
                self.client_address[0].replace(".", "_"),
                model,
                "input",
            )
            input_path = os.path.join(input_folder, "test.jpg")

            if not os.path.exists(input_folder):
                os.makedirs(input_folder)

            image = skimage.io.imread(base64.b64decode(data["image"]), plugin="imageio")
            skimage.io.imsave(input_path, image)

            self.send(
                json.dumps({"response": "Message", "message": "Running model"}).encode(
                    "utf-8"
                )
            )
            prediction = models[model].detect([image])[0]
            prediction["response"] = f"Results"

            for key in prediction:
                prediction[key] = prediction[key].tolist()
            data = json.dumps(prediction).encode("utf-8")

            self.send(
                json.dumps(
                    {"response": "Message", "message": f"Receiving 0KiB"}
                ).encode("utf-8")
            )

            self.send(data)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


def client(ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(bytes(message, "ascii"))
        response = str(sock.recv(1024), "ascii")
        print("Received: {}".format(response))


if __name__ == "__main__":

    car_parts_config = mrcnn_inference.CarPartsConfigSmallest()
    damage_config = mrcnn_inference.CarDamageConfig()
    models = {
        "parts": mrcnn_inference.MRCNN(
            car_parts_config,
            "Mask_RCNN/models/car_parts_smallest_fixed_anno.h5",
        ),
        "damage": mrcnn_inference.MRCNN(
            damage_config, "Mask_RCNN/models/car_damage.h5"
        ),
    }

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 65432))
    HOST, PORT = s.getsockname()[0], 65432
    print("Server IP", HOST)
    s.close()

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        ip, port = server.server_address
        print(ip, port)
        server.serve_forever()
