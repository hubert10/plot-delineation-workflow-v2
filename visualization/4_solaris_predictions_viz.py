#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on FEB 02/02 at Manobi Africa/ ICRISAT 

@Contributors: 
          Pierre C. Traore - ICRISAT/ Manobi Africa
          Steven Ndung'u' - ICRISAT/ Manobi Africa
          Joel Nteupe - Manobi Africa
          John bagiliko - ICRISAT Intern
          Rosmaelle Kouemo - ICRISAT Intern
          Hubert Kanyamahanga - ICRISAT/ Manobi Africa
          Glorie Wowo -  ICRISAT/ Manobi Africa
"""
import sys
sys.path.append("./")
import os
import cv2
import skimage
import pathlib
import rasterio
import solaris as sol
import matplotlib.pyplot as plt
from utils.config import PROJECT_ROOT
from solaris.vector import mask
from rasterio.plot import show
from utils.config import roi_image

# Get the project root directory
project_path = PROJECT_ROOT
RCNN_ROOT = os.path.abspath(project_path + "Mask_RCNN")
os.chdir(RCNN_ROOT)
print("Printing the current project root dir".format(os.getcwd()))

input_raster = PROJECT_ROOT + "results/inputs/"
filtered_jpg = PROJECT_ROOT + "results/predicted/"
geo_raster = PROJECT_ROOT + "results/georeferenced/"

input_raster = pathlib.Path(input_raster + roi_image.split(".")[0])
filtered_jpg = pathlib.Path(filtered_jpg + roi_image.split(".")[0])
geo_raster = pathlib.Path(geo_raster + roi_image.split(".")[0])

ref_image = skimage.io.imread(os.path.join(input_raster, roi_image))
mask_image = skimage.io.imread(os.path.join(filtered_jpg, roi_image.split(".")[0] + '.jpg'))
geo_raster = skimage.io.imread(os.path.join(geo_raster, roi_image))

geoms = mask.mask_to_poly_geojson(mask_image, channel_scaling=[1, -1, -1])

fig, (axr, axl, axg) = plt.subplots(1, 3, figsize=(25, 9))
ref_image = ref_image.swapaxes(2, 0)
ref_image = ref_image.swapaxes(2, 1)
show(ref_image, ax=axr, title="Testing Image")

mask_image = mask_image.swapaxes(2, 0)
mask_image = mask_image.swapaxes(2, 1)
show(mask_image, ax=axl, title="Pred. with Smooth Blending")

show(geo_raster, ax=axg, title="Pred. with Smooth Blending and Solaris")

plt.show()

# Revert the mask to the original crs and affine tranformation for matching.
result_polys = sol.vector.polygon.georegister_px_df(
    geoms, affine_obj=ref_image.transform, crs=ref_image.crs
)
