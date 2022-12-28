#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Steven Ndung'u' and Hubert K

"""
### Geo-referenced to shapefiles and geojsons

# The aim of this script is to convert predicted
# raster polygones saved into exploitable shapefiles
# using Solaris package as one of its utility is for
# Converting between geospatial raster and vector 
# formats and machine learning-compatible formats


import sys
sys.path.append("./")
import os
import pathlib
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

input_raster_p = PROJECT_ROOT + "results/inputs/"
input_raster_p = input_raster_p + roi_image.split(".")[0]
input_raster_p = pathlib.Path(input_raster_p)

geo_raster_p = PROJECT_ROOT + "results/georeferenced/"
geo_raster_p = geo_raster_p + roi_image.split(".")[0]
geo_raster_p = pathlib.Path(geo_raster_p)

input_raster = rasterio.open(os.path.join(input_raster_p, roi_image))
geo_raster = skimage.io.imread(os.path.join(geo_raster_p, roi_image))

geoms = mask.mask_to_poly_geojson(pred_arr=geo_raster, reference_im=input_raster)
geoms.crs = "EPSG:4326"

result_polys = sol.vector.polygon.georegister_px_df(
    geoms, affine_obj=input_raster.transform, crs=input_raster.crs
)
# Save Output to the shape files
# Create dir for saving predictions
dir_output = PROJECT_ROOT + "results/savedfiles/"
output_dir = create_dir(dir_output + "/" + roi_image.split(".")[0])

os.chdir(output_dir)
result_polys.to_file(
    "{}{}".format(roi_image.split(".")[0], ".geojson"), driver="GeoJSON"
)
result_polys.to_file("{}{}".format(roi_image.split(".")[0], ".shp"))


# See the comments below for the next step of this prediction
#########################################################################################
#              geospatial raster and vector formats
# 
#              run: python visualization/4_solaris_predictions_viz.py
#########################################################################################
