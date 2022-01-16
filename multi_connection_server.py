import json
import socket
import os
from _thread import *
import mrcnn_inference

def read_message(conn):
    data = conn.recv(4)
    if data:
        msg_len = int.from_bytes(data, "big")
        print("Message size", msg_len)
        return conn.recv(msg_len)


def send_message(conn, message):
    conn.send(len(message).to_bytes(4, "big"))
    conn.sendall(message)


def client_thread(conn, addr):

    try:
        while True:
            model_type = read_message(conn)
            print(model_type)
            data = conn.recv(4)
            if data:
                msg_len = int.from_bytes(data, "big")
                print("Client image size", msg_len)
                filename = 'img_{}.jpg'.format(''.join(addr[0].split('.')))
                f = open(filename, 'wb')
                for i in range(msg_len):
                    data = conn.recv(1)
                    if not data:
                        break
                    f.write(data)

                f.close()

                send_message(conn, json.dumps(
                    {"response": "Downloaded"}).encode('utf-8'))

                if model_type == b'damage':
                    results = damage_model.detect(filename)
                elif model_type == b'parts':
                    results = car_parts_model.detect(filename)

                filename_out = 'img_{}_out.jpg'.format(''.join(addr[0].split('.')))

                if model_type == b'damage':
                    damage_model.visualize(results, filename_out, mrcnn_inference.class_names_car_damage)
                elif model_type == b'parts':
                    car_parts_model.visualize(results, filename_out, mrcnn_inference.class_names_car_parts_smallest)

                r = results[0]
                r.pop('masks')
                for key in r:
                    r[key] = r[key].tolist()
                r["response"] = "NoMasksResults"

                result = json.dumps(r).encode('utf-8')

                print("Result size", len(result))
                send_message(conn, json.dumps(
                    {"response": "Sending", "size": len(result)}).encode('utf-8'))

                send_message(conn, result)
                print("Sent results")

                filesize = os.path.getsize(filename_out)
                print('Out file size', filesize)
                conn.send(filesize.to_bytes(4, 'big'))

                f = open(filename_out, 'rb')
                for i in range(filesize):
                    conn.send(f.read(1))
                f.close()

                print("Sent out image")
            else:
                break
    except ConnectionResetError as e:
        print('exception occured', e)
    print('exit', addr)


try:
    PORT = 65432
    print("Socket connect")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", PORT))
    HOST = s.getsockname()[0]
    print("Socket connected", HOST)
    s.close()

    car_parts_config = mrcnn_inference.CarPartsConfigSmallest()
    car_parts_model = mrcnn_inference.MRCNN('inference', car_parts_config, "Mask_RCNN/models/car_parts_smallest_fixed_anno.h5")

    damage_config = mrcnn_inference.CarDamageConfig()
    damage_model = mrcnn_inference.MRCNN('inference', damage_config, "Mask_RCNN/models/car_damage.h5")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((HOST, PORT))
        print(s.getsockname())
    except socket.error as e:
        print(str(e))

    print('Waiting for a connection...')
    s.listen()

    while True:
        conn, addr = s.accept()
        print('Connected by', addr)
        start_new_thread(client_thread, (conn, addr,))
except KeyboardInterrupt:
    s.close()
