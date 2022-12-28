"""
This script hepls in cleaning grayscale image to whitten
pixels inside the plot image and darken other parts
"""

import sys

sys.path.append("./")
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import os
import numpy as np
import skimage
import pathlib
from os import listdir
from os.path import isfile
from tkinter import *
from tkinter import Tcl
import skimage.io as io
from utils.config import PROJECT_ROOT
from scipy import ndimage
from utils.config import roi_image
from utils.make_dir import create_dir

# Get the project root directory
project_path = PROJECT_ROOT
os.chdir(PROJECT_ROOT)


def whitten_image_darken_else_filtering(input_img_path, save_path):
    """
    This function is used to filter the image directly after
    it is out from the deep learning predictions
    """
    nlyTIFF = [
        os.path.join(input_img_path, f)
        for f in listdir(input_img_path)
        if isfile(os.path.join(input_img_path, f)) and f.endswith(".jpg")
    ]
    nlyTIFF = Tcl().call("lsort", "-dict", nlyTIFF)
    if len(nlyTIFF) >= 2:
        for i in range(len(nlyTIFF)):
            grayscale = io.imread(nlyTIFF[i], plugin="matplotlib")
            median_filtered = ndimage.median_filter(grayscale, size=3)
            threshold = skimage.filters.threshold_li(median_filtered)
            predicted = np.uint8(median_filtered > threshold) * 255
            io.imsave(
                os.path.join(
                    save_path, "{}.{}".format(input_img_path.split("/")[-1], "jpg")
                ),
                predicted,
            )

    elif len(nlyTIFF) == 1:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = 1000000000
        grayscale = io.imread(nlyTIFF[0], plugin="matplotlib")
        median_filtered = ndimage.median_filter(grayscale, size=3)
        threshold = skimage.filters.threshold_li(grayscale)
        predicted = np.uint8(median_filtered > threshold) * 255
        io.imsave(
            os.path.join(
                save_path, "{}.{}".format(input_img_path.split("/")[-1], "jpg")
            ),
            predicted,
        )
    else:
        print("No jpg file given in the path provided")


new_tif = roi_image.split(".")[0]
input_path = PROJECT_ROOT + "results/predicted/" + new_tif

# Path to save the outputs
save_path = PROJECT_ROOT + "results/filtered/"
# Create dir for saving predictions
output_dir = create_dir(save_path + roi_image.split(".")[0])
save_path = pathlib.Path(output_dir)

# Function call
whitten_image_darken_else_filtering(input_path, save_path)


# See the comments below for the next step of this post-preprocessing
#######################################################################################################################
#                      Georeferencing of the images generated.
#                      run: python 2_geo_referencing.py
#######################################################################################################################
