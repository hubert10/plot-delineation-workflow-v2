import os, sys
from Mask_RCNN.mrcnn.config import Config

##########################################################################################################################
#                     configurations                                                                                     #
##########################################################################################################################

# Set configurations depending on the machine capacity you are using.

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = BASE_DIR + "/"
print(PROJECT_ROOT)


class CustomConfig(Config):
    def __init__(self, num_classes):

        if num_classes > 1:
            raise ValueError(
                "{} classes were found. This is version supports 1 class"
                " continue the training.".format(num_classes)
            )

        self.NUM_CLASSES = num_classes + 1  # Background + (Plot)
        super().__init__()

    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 1  # background + 1 (plot)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 30  # TODO, this needs to be increased to 500 or something when you have high machine capacity

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 15

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


# Define ROI Raster
roi_image = "tile_4096_4096.tif"
# roi_image = "debi_tiguet_image.tif"
# roi_image = "mali.tif"
