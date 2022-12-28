import os
from os import listdir
from os.path import join
from os.path import isfile
from tkinter import Tcl
import rasterio as rio
from rasterio.merge import merge

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = BASE_DIR + "/"
input_path = PROJECT_ROOT + "Data/tile/"
output_path = PROJECT_ROOT + "samples/roi/couche_gamma.tif"
print(PROJECT_ROOT)


def merge_multiple_rasters(input_path, output_path):
    """
    Merges multiple rasters tiles into one unified Raster Image
    """
    nlyTIFF = [
        os.path.join(input_path, f)
        for f in listdir(input_path)
        if isfile(join(input_path, f)) and f.endswith(".tif")
    ]
    nlyTIFF = Tcl().call("lsort", "-dict", nlyTIFF)
    # List for the source files
    src_files_to_mosaic = []
    # Iterate over raster files and add them to source -list in 'read mode'
    for fp in nlyTIFF:
        src = rio.open(fp)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    # Copy the metadata
    out_meta = src.meta.copy()
    # Update the metadata
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        }
    )
    with rio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)


# Function Call
merge_multiple_rasters(input_path, output_path)
