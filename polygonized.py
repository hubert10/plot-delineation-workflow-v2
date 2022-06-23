import numpy as np
from shapely.geometry import shape
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
from affine import Affine
from skimage.morphology import square, erosion, dilation
import os
from tqdm.auto import tqdm
from osgeo import ogr
from osgeo import osr, gdal


class Polygonized:
    def mask_to_raster(input_path, output_shp):
        """Transforme mask into vector file(Geojson) .

        Arguments
        ---------
        input_path : Str
            Pathto the image .
        output_path : Str
            Path to the output produced .

        Returns
        -------
        A vector file of the raster.

        Notes
        -----
        This functions depends on "gdal" and its subfunctions.
        """
        # name='/home/nteupe/building/mask.tif'
        open_image = gdal.Open(input_path)
        input_band = open_image.GetRasterBand(1)
        # create output data source
        # output_shp = "/home/nteupe/all_results/all/ZZ34"
        shp_driver = ogr.GetDriverByName("ESRI Shapefile")
        prj = open_image.GetProjection()
        # create output file name
        output_shapefile = shp_driver.CreateDataSource(output_shp + ".shp")
        new_shapefile = output_shapefile.CreateLayer(
            output_shp, srs=osr.SpatialReference(wkt=prj)
        )
        gdal.Polygonize(input_band, input_band, new_shapefile, -1, [], callback=None)
        new_shapefile.SyncToDisk()


in_shp_file_name = "/home/hubert/Desktop/Heuristics/plot-delineation-workflow/samples/roi/MaliDec2016/Couche_gamma.shp"
out_raster_file_name = "/home/hubert/Desktop/Heuristics/plot-delineation-workflow/samples/roi/MaliDec2016/Couche_gamma.tif"

Polygonized.mask_to_raster(merge_path, output_shp)
