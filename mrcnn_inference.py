import os
import sys
import skimage.io
import numpy as np

ROOT_DIR = os.path.abspath("./Mask_RCNN/")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize
from mrcnn import model as modellib, utils
from mrcnn.config import Config


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


class CarDamageConfig(Config):
    NAME = "car_damage"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)


class_names = {'parts': ['BG', 'bumper', 'glass', 'door',
                         'light', 'hood', 'mirror', 'trunk', 'wheel'],
               'damage': ['BG', 'damage']}


class MRCNN:
    def __init__(self, mode, config, model_path):
        self.config = config
        self.model = modellib.MaskRCNN(
            mode=mode, model_dir=MODEL_DIR, config=self.config)
        self.model.load_weights(model_path, by_name=True)

    def detect(self, image_path):
        self.image = skimage.io.imread(image_path)
        self.results = self.model.detect([self.image])
        return self.results

    def raw_detect(self, image_path, verbose=0):
        self.image = skimage.io.imread(image_path)
        molded_images, image_metas, windows = self.model.mold_inputs([
                                                                     self.image])

        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        anchors = self.model.get_anchors(image_shape)
        anchors = np.broadcast_to(
            anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        detections, _, _, mrcnn_masks, _, _, _ =\
            self.model.keras_model.predict(
                [molded_images, image_metas, anchors], verbose=0)

        zero_ix = np.where(detections[0, :, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0, 0]

        boxes = detections[0, :N, :4]
        class_ids = detections[0, :N, 4].astype(np.int32)
        scores = detections[0, :N, 5]
        masks = mrcnn_masks[0, np.arange(N), :, :, class_ids]

        window = utils.norm_boxes(windows[0], image_shape[:2])

        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, self.image.shape[:2])

        return {'boxes': boxes.tolist(),  'class_ids': class_ids.tolist(), 'scores': scores.tolist(), 'masks': masks.tolist()}

    def visualize(self, results, filename, class_names):
        r = results[0]
        visualize.display_instances(self.image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], show_img=False, filename=filename)
