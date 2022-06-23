import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Set this to True to see more logs details
os.environ["AUTOGRAPH_VERBOSITY"] = "5"
tf.autograph.set_verbosity(3, False)
tf.cast
import warnings

warnings.filterwarnings("ignore")
from utils.config import CustomConfig

tf.compat.v1.disable_eager_execution()


# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR + "/"
print(PROJECT_ROOT)

# Get the project root directory
project_path = PROJECT_ROOT
RCNN_ROOT = os.path.abspath(project_path + "Mask_RCNN")
os.chdir(RCNN_ROOT)
print("Printing the current project root dir".format(os.getcwd()))

# Import Mask RCNN
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.model import log
from PIL import Image, ImageDraw
import cv2
import os

CLASS_NAMES = [
    "BG",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class SimpleConfig(Config):
    NAME = "coco_inference"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)


model = modellib.MaskRCNN(
    mode="inference", config=SimpleConfig(), model_dir=os.getcwd()
)

model.load_weights(filepath="mask_rcnn_coco.h5", by_name=True)

image = cv2.imread(PROJECT_ROOT + "sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = model.detect([image], verbose=0)

r = r[0]

visualize.display_instances(
    image=image,
    boxes=r["rois"],
    masks=r["masks"],
    class_ids=r["class_ids"],
    class_names=CLASS_NAMES,
    scores=r["scores"],
)
