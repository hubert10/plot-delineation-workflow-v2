import cv2
import os
from datetime import datetime
from skimage import io
import skimage.io as io
from utils.config import PROJECT_ROOT
from PIL import Image
import numpy as np
from skimage.color import rgb2gray, gray2rgb


# Get the project root directory
project_path = PROJECT_ROOT
RCNN_ROOT = os.path.abspath(project_path)
os.chdir(RCNN_ROOT)
path_to_image = PROJECT_ROOT + "results/Test/geo_referenced/gray_debi_tiguet_image.tif"
end_time = datetime.now()
now = datetime.now()  # current date and time
time = now.strftime("%m%d%Y_%H%M")
img = cv2.imread(path_to_image)  # GRAY
img = gray2rgb(img)
io.imsave(
    os.path.join(
        PROJECT_ROOT + "results/Test/geo_referenced",
        "RGB_from_{}{}{}".format(
            str(path_to_image).split("/")[-1].split("/")[0], str(time), ".tif"
        ),
    ),
    img,
)
