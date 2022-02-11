import mrcnn_inference
from mrcnn import utils
import json
import socket
import os
import base64
from _thread import *

# ROOT_DIR = os.path.abspath("./Mask_RCNN/")
# sys.path.append(ROOT_DIR)  # To find local version of the library


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


class ClientData:
    def __init__(self, ip, model):
        self.dirname = os.path.join(
            'results', ip.replace('.', '_'), model)
        self.dirname_in = os.path.join(
            self.dirname, 'input')
        self.dirname_out = os.path.join(
            self. dirname, 'output')

        if not os.path.isdir(self.dirname_in):
            os.makedirs(self.dirname_in)
        if not os.path.isdir(self.dirname_out):
            os.makedirs(self.dirname_out)

        self.annotations_path = os.path.join(self.dirname, 'annotations.json')

    def current_id(self):
        return len(os.listdir(self.dirname_in))


class Image:
    def __init__(self, name, folder):
        self.name = name
        self.path = os.path.join(folder, name)

    def size(self):
        return human_readable_size(
            os.path.getsize(self.path))


def human_readable_size(size, decimal_places=0):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f'{size:.{decimal_places}f} {unit}'


def client_thread(conn, addr):
    try:
        while True:
            client_msg = read_message(conn)
            if client_msg is None:
                break
            client_msg = json.loads(client_msg)

            client_data = ClientData(addr[0], client_msg['model'])

            image_id = client_data.current_id()
            input_image = Image(f'{image_id}.jpg', client_data.dirname_in)

            if client_msg['annotations'] is not None:
                image_annotations = client_msg['annotations']

                if os.path.exists(client_data.annotations_path):
                    with open(client_data.annotations_path, 'r') as anno_file:
                        annotations_data = json.load(anno_file)
                else:
                    annotations_data = {'annotations': [], 'images': []}

                last_anno_id = len(annotations_data['annotations'])
                for annotation in image_annotations:
                    annotation['image_id'] = image_id
                    annotation['id'] = last_anno_id
                    last_anno_id += 1
                    annotations_data['annotations'].append(annotation)
                annotations_data['images'].append(
                    {'file_name': input_image.name, 'id': image_id})
                with open(client_data.annotations_path, 'w') as anno_file:
                    json.dump(annotations_data, anno_file,
                              indent=4, sort_keys=True)

            with open(input_image.path, 'wb') as image_file:
                image_file.write(base64.b64decode(client_msg['image']))

            send_message(conn, json.dumps(
                {'response': 'Message', 'message': 'Running model'}).encode('utf-8'))

            print(client_msg['type'])
            if client_msg['type'] == 'raw':
                r = models[client_msg['model']].raw_detect(
                    input_image.path)

            elif client_msg['type'] == 'image':

                results = models[client_msg['model']].detect(input_image.path)
                if not results[0]['rois'].shape[0]:
                    send_message(conn, json.dumps(
                        {'response': 'No results'}).encode('utf-8'))
                    print('no results')
                    continue

                output_image = Image(f'{image_id}_out.jpg',
                                     client_data.dirname_out)

                models[client_msg['model']].visualize(
                    results, output_image.path, mrcnn_inference.class_names[client_msg['model']])

                r = results[0]
                r.pop('masks')
                for key in r:
                    r[key] = r[key].tolist()

                print('Out file size', output_image.size())

                with open(output_image.path, 'rb') as image_file:
                    r['image'] = base64.b64encode(
                        image_file.read()).decode('utf-8')

            r['response'] = f'Results {client_msg["type"]}'

            result = json.dumps(r).encode('utf-8')

            print('Result size', len(result))
            send_message(conn, json.dumps(
                {'response': 'Message', 'message': f'Receiving {human_readable_size(len(result))}'}).encode('utf-8'))

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
