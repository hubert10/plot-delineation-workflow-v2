"""
@author: Steven Ndung'u' and Hubert K

"""
### Georeference Masks

# Example found in the workstation workflow
# The aim of this script is to convert prediction patches saved
# in jp, png, etc to TIF format referencing the metadata of the
# original satelite imagery

import sys, os

sys.path.append("./")
import rasterio as rio
import pathlib
from utils.make_dir import create_dir
from utils.config import PROJECT_ROOT
from utils.config import roi_image

#### Convert the PNG predictions to Rasters Tif format using the original image

# converts from png to tiff


def convert_jpg_to_tif_and_save(input_img_jpg, save_path, georef_img_tif, idx):
    # Input jpg/png image, to convert as geotiff
    img = rio.open(str(input_img_jpg))
    img = img.read(1)
    # Input image for coordinate reference
    with rio.open(
        str(georef_img_tif)
        # + "/"
        # + str(input_img_jpg).replace("jpg", "tif").split("/")[-1]
    ) as naip:
        # open georeferenced.tif for writing

        with rio.open(
            str(save_path)
            + "/"
            + "{}.tif".format(str(input_img_jpg).split("/")[-1].split(".")[0]),
            "w",
            driver="GTiff",
            count=1,
            height=img.shape[0],
            width=img.shape[1],
            dtype=img.dtype,
            crs=naip.crs,
            transform=naip.transform,
        ) as dst:
            print(dst.crs)
            dst.write(img, indexes=1)

    ############## Set the paths #############


# path of png or jpg image predicted from smoothing algorithm
input_img_jpg = PROJECT_ROOT + "results/filtered/"
input_img_jpg = input_img_jpg + roi_image.split(".")[0]
input_img_jpg = pathlib.Path(input_img_jpg)

# folder with original raster image (from original tif)
# Put the raster image you are trying to run predictions in this folder
# Example: tile_4096_4096.tif
georef_img_tif = PROJECT_ROOT + "results/inputs/"
georef_img_tif = pathlib.Path(georef_img_tif + roi_image.split(".")[0])
georef_img_tif = os.path.join(georef_img_tif, roi_image)
print(georef_img_tif)
# Path to save the outputs
save_path = PROJECT_ROOT + "results/georeferenced/"
# Create dir for saving predictions
output_dir = create_dir(save_path + roi_image.split(".")[0])
save_path = pathlib.Path(output_dir)
input_img_jpg, georef_img_tif, save_path

# Import the images, convert them to tif and save back in defined folder

images = list(input_img_jpg.glob("*"))

for idx, val in enumerate(images):
    convert_jpg_to_tif_and_save(str(val), save_path, georef_img_tif, idx)


# See the comments below for the next step of this prediction
#######################################################################################################################
#                      geospatial raster and vector formats
#                      run: python 3_solaris_raster_to_vector.py
#######################################################################################################################
