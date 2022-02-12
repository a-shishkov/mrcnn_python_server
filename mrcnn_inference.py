import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath("./Mask_RCNN/")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model
from mrcnn.config import Config
from mrcnn import utils


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


class_names = {
    "parts": [
        "BG",
        "bumper",
        "glass",
        "door",
        "light",
        "hood",
        "mirror",
        "trunk",
        "wheel",
    ],
    "damage": ["BG", "damage"],
}


class MRCNN(model.MaskRCNN):
    def __init__(self, config, model_path):
        super().__init__(mode="inference", model_dir=MODEL_DIR, config=config)
        self.load_weights(model_path, by_name=True)

    def detect(self, images, verbose=0):
        assert (
            len(images) == self.config.BATCH_SIZE
        ), "len(images) must be equal to BATCH_SIZE"

        if verbose:
            model.log("Processing {} images".format(len(images)))
            for image in images:
                model.log("image", image)

        molded_images, image_metas, windows = self.mold_inputs(images)

        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert (
                g.shape == image_shape
            ), "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            model.log("molded_images", molded_images)
            model.log("image_metas", image_metas)
            model.log("anchors", anchors)

        detections, _, _, mrcnn_mask, _, _, _ = self.keras_model.predict(
            [molded_images, image_metas, anchors], verbose=0
        )

        results = []
        for i, image in enumerate(images):
            (
                final_rois,
                final_class_ids,
                final_scores,
                final_masks,
            ) = self.unmold_detections(
                detections[i],
                mrcnn_mask[i],
                image.shape,
                molded_images[i].shape,
                windows[i],
            )
            results.append(
                {
                    "boxes": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                    "masks": final_masks,
                }
            )
        return results

    def unmold_detections(
        self, detections, mrcnn_mask, original_image_shape, image_shape, window
    ):
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0
        )[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        return [
            boxes.tolist(),
            class_ids.tolist(),
            scores.tolist(),
            masks.tolist(),
        ]
