import base64
import json
import os
import socket
import threading
import socketserver
from cv2 import sort

import numpy as np
import mrcnn_inference
from mrcnn_inference import utils
import skimage.io
from client import client
from pycocotools.coco import COCO


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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

    def demo(self):
        demo_coco = COCO(os.path.join("demo_image_coco.json"))

        image_filename = demo_coco.loadImgs(1)[0]["file_name"]
        image = skimage.io.imread(image_filename)
        with open(image_filename, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")

        classes = {}
        predictions = {}
        for model in models:
            class_names = []
            for cat in demo_coco.loadCats(demo_coco.getCatIds(supNms=model)):
                class_names.append(cat["name"])
            classes[model] = class_names

            anns = demo_coco.loadAnns(
                demo_coco.getAnnIds(catIds=demo_coco.getCatIds(supNms=model))
            )

            boxes = []
            class_ids = []
            scores = []
            masks = []
            for ann in anns:
                box = [
                    ann["bbox"][1],
                    ann["bbox"][0],
                    ann["bbox"][1] + ann["bbox"][3],
                    ann["bbox"][0] + ann["bbox"][2],
                ]
                boxes.append(box)
                class_ids.append(ann["category_id"])
                scores.append(1.0)
                masks.append(demo_coco.annToMask(ann)[box[0] : box[2], box[1] : box[3]])
            predictions[model] = {
                "boxes": np.array(boxes),
                "class_ids": np.array(class_ids),
                "scores": np.array(scores),
                "masks": np.array(masks),
            }

        return predictions, image, encoded, classes

    def predict(self, data):
        image = skimage.io.imread(base64.b64decode(data["image"]), plugin="imageio")
        # skimage.io.imsave(input_path, image)

        self.send(
            json.dumps({"response": "Message", "message": "Running model"}).encode(
                "utf-8"
            )
        )

        predictions = {}
        for model in models:
            predictions[model] = models[model].detect([image])[0]

        return predictions, image

    def save_anno(self, image, data):
        client_dir = os.path.join("results", self.client_address[0].replace(".", "_"))
        images_dir = os.path.join(client_dir, "images")

        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        anno_path = os.path.join(client_dir, "annotations.json")
        if os.path.exists(anno_path):
            with open(anno_path, "r") as anno_file:
                anno_data = json.load(anno_file)
        else:
            anno_data = {"annotations": [], "images": []}

        image_id = len(os.listdir(images_dir))
        skimage.io.imsave(os.path.join(images_dir, f"{image_id}.jpg"), image)

        last_anno_id = len(anno_data["annotations"])
        for anno in data["annotations"]:
            anno["image_id"] = image_id
            anno["id"] = last_anno_id
            last_anno_id += 1
            anno_data["annotations"].append(anno)
        anno_data["images"].append(
            {"file_name": f"images/{image_id}.jpg", "id": image_id}
        )
        with open(anno_path, "w") as anno_file:
            json.dump(anno_data, anno_file, indent=4, sort_keys=True)

    def handle(self):
        while True:
            data = self.read()

            if data is None:
                return

            data = json.loads(data)

            isDemo = ("demo", True) in data.items()
            if isDemo:
                predictions, image, encoded, classes = self.demo()
            else:
                predictions, image = self.predict(data)
                classes = {
                    "parts": mrcnn_inference.class_names["parts"],
                    "damage": mrcnn_inference.class_names["damage"],
                }

            if "annotations" in data and data["annotations"] is not None:
                self.save_anno(image, data)

            for model in models:
                if predictions[model]["boxes"].size == 0:
                    self.send(json.dumps({"response": "No results"}).encode("utf-8"))
                    continue

            intersections = [[] for _ in range(predictions["parts"]["boxes"].shape[0])]

            for i in range(predictions["damage"]["boxes"].shape[0]):
                if isDemo:
                    y1, x1, y2, x2 = predictions["damage"]["boxes"][i]
                    damage_mask = np.zeros(image.shape[:2], dtype=np.bool)
                    damage_mask[y1:y2, x1:x2] = predictions["damage"]["masks"][i]
                else:
                    damage_mask = utils.unmold_mask(
                        predictions["damage"]["masks"][i],
                        predictions["damage"]["boxes"][i],
                        image.shape,
                    )
                for j in range(predictions["parts"]["boxes"].shape[0]):
                    if isDemo:
                        y1, x1, y2, x2 = predictions["parts"]["boxes"][j]
                        parts_mask = np.zeros(image.shape[:2], dtype=np.bool)
                        parts_mask[y1:y2, x1:x2] = predictions["parts"]["masks"][j]
                    else:
                        parts_mask = utils.unmold_mask(
                            predictions["parts"]["masks"][j],
                            predictions["parts"]["boxes"][j],
                            image.shape,
                        )

                    if np.count_nonzero(np.logical_and(damage_mask, parts_mask)) > 0:
                        intersections[j].append(i)

            self.send(
                json.dumps({"response": "Message", "message": f"Receiving"}).encode(
                    "utf-8"
                )
            )

            send_data = {
                "response": "Results",
                "predictions": predictions,
                "intersections": intersections,
            }
            if isDemo:
                send_data["image"] = encoded
                send_data["classes"] = classes

            self.send(json.dumps(send_data, cls=NumpyEncoder).encode("utf-8"))


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


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
    s.close()

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        ip, port = server.server_address
        print(ip, port)

        # server.serve_forever()

        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        print("Server loop running in thread:", server_thread.name)

        # client(ip, port, "image64.jpg")

        # server.shutdown()
        server_thread.join()
