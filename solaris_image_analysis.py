#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 19:14:42 2022

@author: hubert
"""

import os
import skimage
import rasterio
import solaris as sol
from utils.config import PROJECT_ROOT
from solaris.vector import mask
from utils.make_dir import create_dir
from utils.config import roi_image

# Get the project root directory
project_path = PROJECT_ROOT
RCNN_ROOT = os.path.abspath(project_path + "Mask_RCNN")
os.chdir(RCNN_ROOT)
print("Printing the current project root dir".format(os.getcwd()))

input_raster = PROJECT_ROOT + "results/Test/inputs/debi_tiguet_image.tif"
geo_raster = PROJECT_ROOT + "results/Test/georeferenced/debi_tiguet_image.tif"

input_raster = rasterio.open(input_raster)
geo_raster = skimage.io.imread(geo_raster)

geoms = mask.mask_to_poly_geojson(pred_arr=geo_raster, reference_im=input_raster)

result_polys = sol.vector.polygon.georegister_px_df(
    geoms, affine_obj=input_raster.transform, crs=input_raster.crs
)
print(result_polys.head())

# Save Output to the shape files
# Create dir for saving predictions
dir_output = PROJECT_ROOT + "results/Test/savedfiles"
output_dir = create_dir(dir_output + "/" + roi_image.split(".")[0])

os.chdir(output_dir)
result_polys.to_file(
    "{}{}".format(roi_image.split(".")[0], ".geojson"), driver="GeoJSON"
)
result_polys.to_file("{}{}".format(roi_image.split(".")[0], ".shp"))


# # For Roi Raster Image
# raster_input_roi = PROJECT_ROOT + "results/Test/inputs/tile_4096_4096.tif"
# raster_input = rasterio.open(raster_input_roi)
# geo_raster = skimage.io.imread(raster_input_roi)
# roi_geoms = mask.mask_to_poly_geojson(pred_arr=geo_raster, reference_im=raster_input)

# roi_result_polys = sol.vector.polygon.georegister_px_df(
#     roi_geoms, affine_obj=raster_input.transform, crs=raster_input.crs
# )
# dir_output = PROJECT_ROOT + "results/Test/inputs"
# output_dir = create_dir(dir_output + "/" + roi_image.split(".")[0])

# os.chdir(output_dir)
# roi_result_polys.to_file(
#     "{}{}".format(roi_image.split(".")[0], ".geojson"), driver="GeoJSON"
# )
# roi_result_polys.to_file("{}{}".format(roi_image.split(".")[0], ".shp"))
