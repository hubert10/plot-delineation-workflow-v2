"""
Created on FEB 02/02 at Manobi Africa/ ICRISAT 

@Contributors: Pierre C. Traore - ICRISAT/ Manobi Africa
          Steven Ndung'u' - ICRISAT/ Manobi Africa
          Joel Nteupe - Manobi Africa
          John bagiliko - ICRISAT Intern
          Rosmaelle Kouemo - ICRISAT Intern
          Hubert Kanyamahanga - ICRISAT/ Manobi Africa
          Glorie Wowo -  ICRISAT/ Manobi Africa
"""
import cv2
import json
import os
import PIL
import numpy as np
from datetime import datetime
from skimage import io, color
import matplotlib.pyplot as plt
from osgeo import gdal
import skimage
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
import skimage.io as io
import matplotlib.image as mpimg

# from tkinter import *
from tkinter import Tcl
import tensorflow as tf

tf.cast


from patchify import patchify, unpatchify


import warnings

warnings.filterwarnings("ignore")

from IPython import get_ipython

get_ipython().system("nvidia-smi")


##########################################################################################################################
##########################################################################################################################
#                                Introduction & Set up working directory                                                         #
##########################################################################################################################
##########################################################################################################################

"""
Steps:
    
1. Prepare the training/Validation data set
 For this case, Get satelite imagery of your area of interest. Slice the images into 1024*1024 patches then split them
 into the training and validation folders. 
2. Annotate your images generated using the https://www.makesense.ai/ platform. Save the images in coco format. 
3. Save the .json file in the respective folders after creating it. 
4. Install and download the M-RCNN module from github.
5. Use the script below accrodingly
"""


##########################################################################################################################
#                                      Model setup                                                                     #
##########################################################################################################################


# Use MRCNN for version 2.X tensorflow
#!git clone https://github.com/BupyeongHealer/Mask_RCNN_tf_2.x.git for tf 2.x  #Steven

# installtensorflow 2.3.0 and keras 2.4
#!pip install tensorflow==2.3.0
#!pip install keras==2.4
#!pip install --upgrade h5py==2.10.0


# Import Mask RCNN

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
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
# Root directory of the project
ROOT_DIR = os.path.abspath(
    "C:\\Users\\Steven\\Downloads\\new_automation-20220128T083924Z-001\\new_automation\\Mask_RCNN"
)


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

##########################################################################################################################
#                     configurations                                                                                     #
##########################################################################################################################

# Set configurations depending on the machine capacity you are using.


class CustomConfig(Config):
    def __init__(self, num_classes):

        if num_classes > 1:
            raise ValueError(
                "{} classes were found. This is version supports 1 class"
                " continue the training.".format(num_classes)
            )

        self.NUM_CLASSES = num_classes + 1
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
    STEPS_PER_EPOCH = 30  # TODO, this needs to be increased to 500 or something when you have high machine capacity

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 15

    DETECTION_MIN_CONFIDENCE = 0.9


##########################################################################################################################
#  creating a custom dataset similar to the coco data set                                                               #
##########################################################################################################################


# notbook preferences
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class CustomDataset(utils.Dataset):
    """Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
    See http://cocodataset.org/#home for more information.
    """

    def load_custom(self, annotation_json, images_dir, dataset_type="train"):
        """Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """

        # Load json from file
        print("Annotation json path: ", annotation_json)
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
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """

        # Load json from file
        print("Annotation json path: ", annotation_json)
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
#  creating the model for plot delineation                                                                               #
##########################################################################################################################

# Visualize the data created


def display_image_samples(dataset_train):
    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)

    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# Load the pre-trained model
def load_training_model(config):
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

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
def load_image_dataset(annotation_path, dataset_path, dataset_type):
    dataset_train = CustomDataset()
    dataset_train.load_custom(annotation_path, dataset_path, dataset_type)
    dataset_train.prepare()
    return dataset_train


# load validation dataset
def load_image_dataset_val(annotation_path, dataset_path, dataset_type):
    dataset_train = CustomDataset()
    dataset_train.load_custom_val(annotation_path, dataset_path, dataset_type)
    dataset_train.prepare()
    return dataset_train


"""
Train the model  in two stages:

1. Only the heads. Freeze all the backbone layers and training only the randomly initialized layers 
(i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, 
 pass layers='heads' to the train() function.

2. Fine-tune all layers.Simply pass layers="all to train all layers.

TODO: Update the parameters.
"""
# Train the head branches
# Passing layers="heads" freezes all layers except the head


def train_head(
    model, dataset_train, dataset_val, config, epochs
):  # Removed the model  and added epochs parameter
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=epochs,
        layers="heads",
    )


# treain all layers
def train_all_layers(
    model, dataset_train, dataset_val, config, epochs
):  # Removed the model  and added epochs parameter
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=epochs,
        layers="all",
    )


#######################################################################################################################
#                                        Model training                                                               #
#######################################################################################################################

# Define the data paths

path = "path to the training dataset prepared"
annotations_path = "annotation_path #json file"
dataset_train = load_image_dataset(annotations_path, path, "train")
dataset_val = load_image_dataset_val(annotations_path, path, "val")

# Load and display random samples
display_image_samples(dataset_train)
display_image_samples(dataset_val)


# train both the head and layers
model = load_training_model()
class_number = 1
config = CustomConfig(class_number)

train_head(model, dataset_train, dataset_val, config, epochs=15)
train_all_layers(model, dataset_train, dataset_val, config, epochs=15)

# save the model if need be, when satisfied with the training.
# Typically not needed because callbacks save after every epoch, if you want to save the model manually...use
# model.keras_model.save_weights(model_path)


# Load Configuration
# class_number = 1
# config = CustomConfig(class_number)
# config.display()
# model = load_training_model(config)

# train_head(model, dataset_train, dataset_train, config)

#######################################################################################################################
#                                  Model testing                                                                      #
#######################################################################################################################

class_number = 1

config = CustomConfig(class_number)
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(
    "C:/Users/Steven/Downloads/new_automation-20220128T083924Z-001/new_automation/saved_model/mask_rcnn_object_0015.h5",
    by_name=True,
)


# Apply a trained model on large image

img = cv2.imread(
    r"C:\Users\Steven\Downloads\new_automation-20220128T083924Z-001\new_automation\debi-tiguet_image.tif"
)  # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

patch_size = 1024

SIZE_X = (
    img.shape[1] // patch_size
) * patch_size  # Nearest size divisible by our patch size
SIZE_Y = (
    img.shape[0] // patch_size
) * patch_size  # Nearest size divisible by our patch size
large_img = Image.fromarray(img)
PIL.Image.MAX_IMAGE_PIXELS = 933120000
large_img = large_img.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
# image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
large_img = np.array(large_img)


# Import the smooth tile stitching while stitching.
os.chdir(
    "C:/Users/Steven/Downloads/new_automation-20220128T083924Z-001/new_automation/"
)

from smooth_tiled_predictions import predict_img_with_smooth_windowing


def predict_image(tile_image):

    results = model.detect([tile_image])
    mask_generated = results[0]["masks"]
    masked_img = np.any(mask_generated.astype(np.bool), axis=-1)
    masked_img = np.stack((masked_img,) * 3, axis=-1)
    # convert it to 4D
    # masked_img = np.expand_dims(masked_img, axis=0)
    return masked_img


def func_pred(img_batch_subdiv):
    res = map(
        lambda img_batch_subdiv: predict_image(img_batch_subdiv), img_batch_subdiv
    )
    subdivs = np.array(list(res))
    return subdivs


start_time = datetime.now()

predictions_smooth = predict_img_with_smooth_windowing(
    large_img,
    window_size=1024,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=3,
    pred_func=(func_pred),
)

end_time = datetime.now()

print("Duration: {}".format(end_time - start_time))

predictions_smooth1 = color.rgb2gray(predictions_smooth)


plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title("Testing Image")
plt.imshow(large_img)


predictions_smooth2 = predictions_smooth > 0.01

plt.subplot(222)
plt.title("Prediction with smooth blending")
plt.imshow(predictions_smooth2)


predictions_smooth1 = predictions_smooth1.astype(np.uint8)

io.imsave(
    os.path.join(
        "C:/Users/Steven/Downloads/new_automation-20220128T083924Z-001/new_automation/Output",
        "Pred_tile8.jpg",
    ),
    predictions_smooth,
)

#######################################################################################################################
#                                  Georeferencing of the images generated.                                                                      #
#######################################################################################################################
start = datetime.strptime(date_1, date_format_str)
end = datetime.strptime(date_2, date_format_str)
# get the difference between two dates as timedelta object
diff = end.date() - start.date()
print("Difference between dates in days:")
print(diff.days)

date_format_str = "%m-%d-%Y, %H:%M:%S"
start = datetime.strptime(now, date_format_str)
end = datetime.strptime(now, date_format_str)
# get the difference between two dates as timedelta object
diff = end.date() - start.date()
print("Difference between dates in days:")
print(diff)
