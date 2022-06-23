#!/bin/bash

# Information:
# This script rasterizes one parameter of a given shapefile to match
# the spatial dimensions as well as the array dimensions of
# the clipped Sentinel-2 .tif file.
#
# full documentation of gdal_rasterize on:
# https://gdal.org/programs/gdal_rasterize.html

# variables to be set:
#
# name of the shapefile (excluding .shp)
NAME=Couche_gamma
# NAME=debi-tiguet_image

# rasterized parameter
PARAMETER=FID_1
# uncomment these two lines if you want to get the Klingenberg + Czech area
# NAME=Projektgebiet_Klingenberg_dissolve
# PARAMETER=Name_n

# spatial reference system code of all_bands.tif
COORD_SYS=EPSG:4326
# extents of all_bands.tif
X_MIN=-5.234818943
Y_MIN=-5.143207998
X_MAX=12.135424502
Y_MAX=12.225942770 

# path (doesn't need to be changed)
cd ../..
PATH_GT=${PWD}/Heuristics/plot-delineation-workflow/samples/roi/MaliDec2016

echo "rasterize parameter ${PARAMETER} of shapefile..."
gdal_rasterize -a ${PARAMETER} -a_srs ${COORD_SYS} -te ${X_MIN} ${Y_MIN} ${X_MAX} ${Y_MAX} -tr 10 -10 -l ${NAME} ${PATH_GT}/${NAME}.shp ${PWD}/Heuristics/plot-delineation-workflow/samples/roi/MaliDec2016/${PARAMETER}.tif
