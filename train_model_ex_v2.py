import os
import xml.etree
from numpy import zeros, asarray
import tensorflow as tf
# Set this to True to see more logs details
os.environ["AUTOGRAPH_VERBOSITY"] = "5"
tf.autograph.set_verbosity(3, False)
tf.cast
import warnings

warnings.filterwarnings("ignore")
from utils.config import CustomConfig

# tf.compat.v1.disable_eager_execution()


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

KANGAROO_ROOT = os.path.join(RCNN_ROOT + "/samples/kangaroo/")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(KANGAROO_ROOT, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory to save logs and trained model while training for backup
DEFAULT_LOGS_DIR = os.path.join(KANGAROO_ROOT, "logs")

print(DEFAULT_LOGS_DIR)
class KangarooDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "kangaroo")

        images_dir = dataset_dir + "/images/"
        annotations_dir = dataset_dir + "/annots/"

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            if image_id in ["00090"]:
                continue

            if is_train and int(image_id) >= 150:
                continue

            if not is_train and int(image_id) < 150:
                continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + ".xml"

            self.add_image(
                "dataset", image_id=image_id, path=img_path, annotation=ann_path
            )

    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for box in root.findall(".//bndbox"):
            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        width = int(root.find(".//size/width").text)
        height = int(root.find(".//size/height").text)
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info["annotation"]
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype="uint8")

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index("kangaroo"))
        return masks, asarray(class_ids, dtype="int32")


class KangarooConfig(Config):
    NAME = "kangaroo_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 2

    STEPS_PER_EPOCH = 131


train_set = KangarooDataset()
train_set.load_dataset(dataset_dir="samples/kangaroo", is_train=True)
train_set.prepare()

valid_dataset = KangarooDataset()
valid_dataset.load_dataset(dataset_dir="samples/kangaroo", is_train=False)
valid_dataset.prepare()

kangaroo_config = KangarooConfig()
kangaroo_config.display()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", model_dir=DEFAULT_LOGS_DIR, config=kangaroo_config)

print(COCO_MODEL_PATH)

model.load_weights(
    filepath=COCO_MODEL_PATH,
    by_name=True,
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_mask", "rpn_model"],
)

model.train(
    train_dataset=train_set,
    val_dataset=valid_dataset,
    learning_rate=kangaroo_config.LEARNING_RATE,
    epochs=1,
    layers="heads",
)

model_path = DEFAULT_LOGS_DIR + "Kangaro_mask_rcnn.h5"
model.keras_model.save_weights(model_path)
