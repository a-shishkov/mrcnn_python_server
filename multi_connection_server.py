import json
import socket
import os
import base64
from _thread import *
import mrcnn_inference


def read_message(conn):
    BUFF_SIZE = 4096
    data = conn.recv(4)
    if data:
        msg_len = int.from_bytes(data, 'big')
        data = b''
        print('Message size', msg_len)
        while len(data) < msg_len:
            packet = conn.recv(msg_len-len(data))
            if not packet:
                return None
            data += packet
        return data


def send_message(conn, message):
    conn.send(len(message).to_bytes(4, 'big'))
    conn.sendall(message)


def client_thread(conn, addr):
    try:
        while True:
            client_msg = read_message(conn)
            if client_msg is None:
                break
            client_msg = json.loads(client_msg)
            print(client_msg['annotations'])

            filename = 'img_{}.jpg'.format(''.join(addr[0].split('.')))
            with open(filename, 'wb') as f:
                f.write(base64.b64decode(client_msg['image']))

            model_type = client_msg['model']

            send_message(conn, json.dumps(
                {'response': 'Message', 'message': 'Running model'}).encode('utf-8'))

            results = models[model_type].detect(filename)

            if not results[0]['rois'].shape[0]:
                send_message(conn, json.dumps(
                    {'response': 'No results'}).encode('utf-8'))
                print('no results')
                continue

            filename_out = 'img_{}_out.jpg'.format(''.join(addr[0].split('.')))

            models[model_type].visualize(
                results, filename_out, mrcnn_inference.class_names[model_type])

            r = results[0]
            r.pop('masks')
            for key in r:
                r[key] = r[key].tolist()

            filesize = os.path.getsize(filename_out)
            print('Out file size', filesize)

            with open(filename_out, 'rb') as f:
                img = f.read()

            r['image'] = base64.b64encode(img).decode('utf-8')
            r['response'] = 'Results'

            result = json.dumps(r).encode('utf-8')

            print('Result size', len(result))
            send_message(conn, json.dumps(
                {'response': 'Message', 'message': 'Sending {} bytes'.format(len(result))}).encode('utf-8'))

            # print('img', list(img))
            print('Img size', len(json.dumps(r['image']).encode('utf-8')))
            send_message(conn, result)
            print('Sent results')

    except ConnectionResetError as e:
        print('exception occured', e)
    print('exit', addr)
    return

try:
    PORT = 65432
    print('Socket connect')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', PORT))
    HOST = s.getsockname()[0]
    print('Socket connected', HOST)
    s.close()

    car_parts_config = mrcnn_inference.CarPartsConfigSmallest()

    damage_config = mrcnn_inference.CarDamageConfig()

    models = {'parts': mrcnn_inference.MRCNN('inference',
                                             car_parts_config,
                                             'Mask_RCNN/models/car_parts_smallest_fixed_anno.h5'),
              'damage': mrcnn_inference.MRCNN('inference',
                                              damage_config,
                                              'Mask_RCNN/models/car_damage.h5')}

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
