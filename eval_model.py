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
# Directory to save logs and trained model
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_model")
# Import Mask RCNN
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.model import log
from Mask_RCNN.mrcnn.model import mold_image
from Mask_RCNN.mrcnn.utils import compute_ap
from Mask_RCNN.mrcnn.model import load_image_gt
from numpy import expand_dims
from numpy import mean
from PIL import Image, ImageDraw

with open("mrcnn/model.py") as f:
    model_file = f.read()

with open("mrcnn/model.py", "w") as f:
    model_file = model_file.replace(
        "self.keras_model = self.build(mode=mode, config=config)",
        "self.keras_model = self.build(mode=mode, config=config)\n        self.keras_model.metrics_tensors = []",
    )
    f.write(model_file)

"""
Set up logging and pre-trained model paths
This will default to sub-directories in your mask_rcnn_dir, but if you want them somewhere else, updated it here.

It will also download the pre-trained coco model.
"""
# Directory to save logs and trained model while training for backup
DEFAULT_LOGS_DIR = os.path.join(RCNN_ROOT, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(RCNN_ROOT, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

##########################################################################################################################
#                     configurations                                                                                     #
##########################################################################################################################

# Set configurations depending on the machine capacity you are using.


class CustomDataset(utils.Dataset):
    """Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
    See http://cocodataset.org/#home for more information.
    """

    def load_custom_train(self, annotation_json, images_dir, dataset_type="train"):
        """Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """

        # Load json from file
        print("Annotation json path for train.: ", annotation_json)
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json["categories"]:
            class_id = category["id"]

            class_name = category["name"]
            if class_id < 1:
                print(
                    'Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                        class_name
                    )
                )
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        # annotations= []
        for annotation in coco_json["annotations"]:
            annotation["category_id"] = 1
            image_id = annotation["image_id"]
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}

        # Split the dataset, if train, get 90%, else 10%
        len_images = len(coco_json["images"])
        if dataset_type == "train":
            img_range = [int(len_images / 9), len_images]
        else:
            img_range = [0, int(len_images / 9)]

        for i in range(img_range[0], img_range[1]):
            image = coco_json["images"][i]
            image_id = image["id"]
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image["file_name"]
                    image_width = image["width"]
                    image_height = image["height"]
                except KeyError as key:
                    print(
                        "Warning: Skipping image (id: {}) with missing key: {}".format(
                            image_id, key
                        )
                    )

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations,
                )

    def load_custom_val(self, annotation_json, images_dir, dataset_type="val"):
        """Load the coco-like dataset from json
        Args:
            annotation_json: The path to the train annotations json file in coco format
            images_dir: The directory holding the images referred to by the json file
        """

        # Load json from file
        print("Annotation json path for val.: ", annotation_json)
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json["categories"]:
            class_id = category["id"]

            class_name = category["name"]
            if class_id < 1:
                print(
                    'Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                        class_name
                    )
                )
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        # annotations= []
        for annotation in coco_json["annotations"]:
            annotation["category_id"] = 1
            image_id = annotation["image_id"]
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}

        # Split the dataset, if train, get 90%, else 10%
        len_images = len(coco_json["images"])
        if dataset_type == "val":
            img_range = [int(len_images / 9), len_images]
        else:
            img_range = [0, int(len_images / 9)]

        for i in range(img_range[0], img_range[1]):
            image = coco_json["images"][i]
            image_id = image["id"]
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image["file_name"]
                    image_width = image["width"]
                    image_height = image["height"]
                except KeyError as key:
                    print(
                        "Warning: Skipping image (id: {}) with missing key: {}".format(
                            image_id, key
                        )
                    )

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations,
                )

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        print(image_info)
        annotations = image_info["annotations"]
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation["category_id"]
            mask = Image.new("1", (image_info["width"], image_info["height"]))
            mask_draw = ImageDraw.ImageDraw(mask, "1")
            for segmentation in annotation["segmentation"]:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        # print("Class_ids, ", class_ids)
        return mask, class_ids

    def count_classes(self):
        class_ids = set()
        for image_id in self.image_ids:
            image_info = self.image_info[image_id]
            annotations = image_info["annotations"]

            for annotation in annotations:
                class_id = annotation["category_id"]
                class_ids.add(class_id)

        class_number = len(class_ids)
        return class_number


##########################################################################################################################
#                 creating the model for plot delineation                                                                               #
##########################################################################################################################

# Visualize the data created


def display_image_samples(dataset_train):
    # Load and display random samples
    image_ids = np.random.choice(2, 4)
    print(dataset_train.image_ids)
    for image_id in image_ids:
        print(image_id)
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# Load the pre-trained model
def load_training_model(config):
    model = modellib.MaskRCNN(
        mode="training", config=config, model_dir=DEFAULT_LOGS_DIR
    )
    print(model)
    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        # print(COCO_MODEL_PATH)
        model.load_weights(
            COCO_MODEL_PATH,
            by_name=True,
            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
        )
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    return model


# load taining image data
def load_image_dataset_train(annotation_path, dataset_path, dataset_type):
    dataset_train = CustomDataset()
    dataset_train.load_custom_train(annotation_path, dataset_path, dataset_type)
    dataset_train.prepare()
    return dataset_train


# load validation image data
def load_image_dataset_val(annotation_path, dataset_path, dataset_type):
    dataset_val = CustomDataset()
    dataset_val.load_custom_val(annotation_path, dataset_path, dataset_type)
    dataset_val.prepare()
    return dataset_val


"""
Train the model  in two stages:

1. Only the heads. Freeze all the backbone layers and training only the randomly initialized layers 
(i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, 
 pass layers='heads' to the train() function.

2. Fine-tune all layers. Simply pass layers="all to train all layers.

TODO: Update the parameters.
"""
# Train the head branches
# Passing layers="heads" freezes all layers except the head


def train_head(
    model, dataset_train, dataset_val, config, epochs
):  # Removed the model and added epochs parameter
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=epochs,
        layers="heads",
    )


# train all layers
def train_all_layers(
    model, dataset_train, dataset_val, config, epochs
):  # Removed the model and added epochs parameter
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=epochs,
        layers="all",
    )


###############################################################################
#                                        Model Evaluation
###############################################################################

# Define the data paths

path_dataset = PROJECT_ROOT + "dataset/train"
annotations_path_train = PROJECT_ROOT + "dataset/annotations.json"
annotations_path_val = PROJECT_ROOT + "dataset/annotations.json"
dataset_train = load_image_dataset_train(annotations_path_train, path_dataset, "train")
dataset_val = load_image_dataset_val(annotations_path_val, path_dataset, "val")
class_number = dataset_train.count_classes()

print("Train: %d" % len(dataset_train.image_ids))
print("Validation: %d" % len(dataset_val.image_ids))
print("Classes: {}".format(class_number))

# TODO: Add Evaluations Metrics

# we define a prediction configuration
class PredictionConfig(Config):
    NAME = "damage"
    NUM_CLASSES = 1 + 1
    DETECTION_MIN_CONFIDENCE = 0.9
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False


# Train both the head and layers
class_number = 1
cfg = PredictionConfig()
cfg.display()
model = modellib.MaskRCNN(mode="inference", config=cfg, model_dir=MODEL_DIR)
# evaluate_model is used to calculate mean Average Precision of the model
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(
            dataset, cfg, image_id
        )
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]
        AP, _, _, _ = compute_ap(
            gt_bbox,
            gt_class_id,
            gt_mask,
            r["rois"],
            r["class_ids"],
            r["scores"],
            r["masks"],
        )

        APs.append(AP)
    mAP = mean(APs)
    # Mean Average Precision
    return mAP


train_mAP = evaluate_model(dataset_train, model, cfg)
print("Train mAP: %.3f" % train_mAP)

# evaluate model on test dataset
test_mAP = evaluate_model(dataset_val, model, cfg)
print("Test mAP: %.3f" % test_mAP)
