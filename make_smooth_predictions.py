"""
Created on FEB 02/02/2022 at Manobi Africa/ ICRISAT 

@Contributors: 
          Pierre C. Traore - ICRISAT/ Manobi Africa
          Steven Ndung'u' - ICRISAT/ Manobi Africa
          Hubert Kanyamahanga - ICRISAT/ Manobi Africa
          Glorie Wowo -  ICRISAT/ Manobi Africa
          Joel Nteupe - Manobi Africa
          John bagiliko - ICRISAT Intern
          Rosmaelle Kouemo - ICRISAT Intern
"""
import os
import PIL
import numpy as np
from datetime import datetime
from skimage import io, color
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.config import CustomConfig, PROJECT_ROOT
from utils.make_dir import create_dir
from utils.config import roi_image

# Set this to True to see more logs details
os.environ["AUTOGRAPH_VERBOSITY"] = "5"
tf.autograph.set_verbosity(3, False)
tf.cast
import warnings

print(tf.executing_eagerly())

warnings.filterwarnings("ignore")
# https://stackoverflow.com/questions/58070174/overcome-opencv-cv-io-max-image-pixels-limitation-in-python
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
# export CV_IO_MAX_IMAGE_PIXELS=1099511627776
import cv2


##########################################################################################################################
##########################################################################################################################
#                                Introduction & Set up working directory                                                 #
##########################################################################################################################
##########################################################################################################################

"""
Steps:
    
1. Load the ROI raster image
 For this case, Get satellite imagery of your area of interest.
2. Convert the raster into  np.array after resizing it to be divisible by our patch_size:1024 for Mask-RCNN 
3. Call the model to detect plot boundaries and return masks as one masked image 
4. Perform local predictions on each tile on overlapping patches using Smoothing Blending Algo, with rotations and mirroring each patch
5. Merge all patches together
"""
##########################################################################################################################
#                                      Model setup                                                                     #
##########################################################################################################################
"""
# Use MRCNN for version 2.X tensorflow
# !git clone https://github.com/BupyeongHealer/Mask_RCNN_tf_2.x.git for tf 2.x  #Steven
# installtensorflow 2.3.0 and keras 2.4
# !pip install tensorflow==2.3.0
# !pip install keras==2.4.0
# !pip install --upgrade h5py==2.10.0
"""

# Get the project root directory
project_path = PROJECT_ROOT
RCNN_ROOT = os.path.abspath(project_path + "Mask_RCNN")
os.chdir(RCNN_ROOT)
print("Printing the current project root dir".format(os.getcwd()))

# Import Mask RCNN
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from PIL import Image

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

# Directory to save logs and trained model
MODEL_DIR = os.path.join(RCNN_ROOT, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(RCNN_ROOT, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

################################################################################################
#                                  Make Predictions                                            #
################################################################################################
class_number = 1

config = CustomConfig(class_number)
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(PROJECT_ROOT + "saved_model/mask_rcnn_object_0030.h5", by_name=True)

# Apply a trained model on large image
# Load Large Image
img = cv2.imread(PROJECT_ROOT + "samples/roi/" + roi_image)  # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Opencv reads images as BGR

print("------------")
print(img)

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
from smooth_tiled_predictions import predict_img_with_smooth_windowing


def predict_image(tile_image):
    """
    Detects plot boundaries and returns
    corresponding masks as one masked image
    """
    results = model.detect([tile_image])
    mask_generated = results[0]["masks"]
    masked_img = np.any(mask_generated.astype(np.bool), axis=-1)
    masked_img = np.stack((masked_img,) * 3, axis=-1)
    return masked_img


def func_pred(img_batch_subdiv):
    res = map(
        lambda img_batch_subdiv: predict_image(img_batch_subdiv), img_batch_subdiv
    )
    subdivs = np.array(list(res))
    return subdivs


now = datetime.now()
start_time = now
starting_time = now.strftime("%m-%d-%Y, %H:%M:%S")

print(
    "+++++++++++++++++++++++++++  Starting Prediction at: {} +++++++++++++++++++++++++++ ".format(
        starting_time
    )
)

predictions_smooth = predict_img_with_smooth_windowing(
    large_img,
    window_size=1024,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=3,
    pred_func=(func_pred),
)
end_time = datetime.now()

print("Duration: {}".format(end_time - start_time))

predictions_smooth_gray = color.rgb2gray(predictions_smooth)

plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title("Testing Image")
plt.imshow(large_img)

plt.subplot(222)
plt.title("Prediction with smooth blending")
plt.imshow(predictions_smooth_gray)
now = datetime.now()  # current date and time
time = now.strftime("%m%d%Y_%H%M")

# Create dir for saving predictions
dir_output = PROJECT_ROOT + "results/predicted/"
output_dir = create_dir(dir_output + roi_image.split(".")[0])

io.imsave(
    os.path.join(
        output_dir, "{}{}{}".format(roi_image.split(".")[0], str(time), ".jpg")
    ),
    predictions_smooth,
)
# See the comments below for the next step of this prediction
#####################################################################
#                      Georeferencing of the images generated.
#                      run: python postpreprocessing/1_image_filtering.py
#
#####################################################################
