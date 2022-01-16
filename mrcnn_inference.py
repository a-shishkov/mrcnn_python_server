import os
import sys
import skimage.io

ROOT_DIR = os.path.abspath("./Mask_RCNN/")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

CAR_PARTS_MODEL_DIR = "/car_parts_models"
# MODEL_PATH = os.path.join(ROOT_DIR, "models/car_parts_smallest_fixed_anno.h5")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class CarPartsConfigSmallest(Config):
    NAME = "car_part"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 9
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

class_names_car_parts_smallest = ['BG', 'bumper', 'glass', 'door',
                                  'light', 'hood', 'mirror', 'trunk', 'wheel']

class CarDamageConfig(Config):
    NAME = "car_damage"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

class_names_car_damage = ['BG', 'damage']

class MRCNN:
    def __init__(self, mode, config, model_path):
        self.config = config
        self.model = modellib.MaskRCNN(mode=mode, model_dir=MODEL_DIR, config=self.config)
        self.model.load_weights(model_path, by_name=True)

    def detect(self, image_path):
        self.image = skimage.io.imread(image_path)
        self.results = self.model.detect([self.image])
        return self.results

    def visualize(self, results, filename, class_names):
        r = results[0]
        visualize.display_instances(self.image, r['rois'], r['masks'], r['class_ids'], 
        class_names, r['scores'], show_img=False, filename=filename)
