#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday May 20 16:31:20 2020

@author: hubert K.
"""
import os
import skimage
import rasterio
from shapely.geometry import shape
import geopandas as gpd
from rasterio import features
import geopandas as gpd
from utils.config import PROJECT_ROOT
from utils.make_dir import create_dir
from utils.config import roi_image

# Get the project root directory
os.chdir(PROJECT_ROOT)
print("Printing the current project root dir".format(os.getcwd()))


def binary_mask_to_poly_geojson(
    imagepath,
    output_path,
    channel_scaling=None,
    reference_im=None,
    cr=None,
    output_type="geojson",
    min_area=27,
    bg_threshold=0,
    simplify=True,
    tolerance=15,
    **kwargs
):
    """Get polygons from an image mask.
    definitions:
     - A binary mask defines a region of interest (ROI) of an image
    Arguments
    ---------
    pred_arr : :class:`numpy.ndarray`
        A 2D array of integers. Multi-channel masks are not supported, and must
        be simplified before passing to this function. Can also pass an image
        file path here.
    channel_scaling : :class:`list`-like, optional
        If `pred_arr` is a 3D array, this argument defines how each channel
        will be combined to generate a binary output. channel_scaling should
        be a `list`-like of length equal to the number of channels in
        `pred_arr`. The following operation will be performed to convert the
        multi-channel prediction to a 2D output ::

            sum(pred_arr[channel]*channel_scaling[channel])

        If not provided, no scaling will be performend and channels will be
        summed.
    reference_im : str, optional
        The path to a reference geotiff to use for georeferencing the polygons
        in the mask. Required if saving to a GeoJSON (see the ``output_type``
        argument), otherwise only required if ``do_transform=True``.
    output_path : str, optional
        Path to save the output file to. If not provided, no file is saved.
    output_type : ``'csv'`` or ``'geojson'``, optional
        If ``output_path`` is provided, this argument defines what type of file
        will be generated - a CSV (``output_type='csv'``) or a geojson
        (``output_type='geojson'``).
    min_area : int, optional
        The minimum area of a polygon to retain. Filtering is done AFTER
        any coordinate transformation, and therefore will be in destination
        units.
    bg_threshold : int, optional
        The cutoff in ``mask_arr`` that denotes background (non-object).
        Defaults to ``0``.
    simplify : bool, optional
        If ``True``, will use the Douglas-Peucker algorithm to simplify edges,
        saving memory and processing time later. Defaults to ``False``.
    tolerance : float, optional
        The tolerance value to use for simplification with the Douglas-Peucker
        algorithm. Defaults to ``0.5``. Only has an effect if
        ``simplify=True``.

    Returns
    -------
    gdf : :class:`geopandas.GeoDataFrame`
        A GeoDataFrame of polygons.

    """
    mask_arr = skimage.io.imread(fname=imagepath)
    if reference_im is None:
        with rasterio.open(imagepath) as ref:
            transform = ref.transform
            crs = ref.crs
            ref.close()
    else:
        with rasterio.open(reference_im) as ref:
            transform = ref.transform
            crs = ref.crs
            ref.close()
    mask = mask_arr > bg_threshold
    mask = mask.astype("uint8")

    polygon_generator = features.shapes(mask_arr, transform=transform, mask=mask)
    polygons = []
    values = []  # pixel values for the polygon in mask_arr
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(0.0)
        if p.area >= min_area:
            polygons.append(shape(polygon).buffer(0.0))
            values.append(value)

    polygon_gdf = gpd.GeoDataFrame(
        {"geometry": polygons, "value": values}, crs=crs.to_wkt()
    )
    if simplify:
        polygon_gdf["geometry"] = polygon_gdf["geometry"].apply(
            lambda x: x.simplify(tolerance=tolerance)
        )
    # changing the crs in case
    if cr is not None:
        polygon_gdf = polygon_gdf.to_crs(epsg=cr)
    # save output files
    if output_path is not None:
        if output_type.lower() == "geojson":
            # if len(polygon_gdf) > 0:
            polygon_gdf.to_file(output_path, driver="GeoJSON")
            # else:
            # save_empty_geojson(output_path, polygon_gdf.crs.to_epsg())
        elif output_type.lower() == "csv":
            polygon_gdf.to_csv(output_path, index=False)
        else:
            polygon_gdf.to_file(output_path)
    print(polygon_gdf)
    return polygon_gdf


input_raster = PROJECT_ROOT + "results/Test/inputs/debi_tiguet_image.tif"
output_dir = PROJECT_ROOT + "results/Test/refs"
output_dir = create_dir(output_dir + "/" + roi_image.split(".")[0] + "/")
os.chdir(output_dir)

# binary_mask_to_poly_geojson(
#     input_raster,
#     output_dir,
#     channel_scaling=None,
#     reference_im=None,
#     cr=None,
#     output_type="geojson",
#     min_area=0.000000032,
#     bg_threshold=0,
#     simplify=True,
#     tolerance=0.00005,
# )


from shapely.geometry import shape
from shapely.geometry import Polygon
import geopandas as gpd
import rasterio
from rasterio import features
import os
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
        open_image = gdal.Open(input_path)
        input_band = open_image.GetRasterBand(1)
        # create output data source
        shp_driver = ogr.GetDriverByName("ESRI Shapefile")
        prj = open_image.GetProjection()
        # create output file name
        output_shapefile = shp_driver.CreateDataSource(output_shp + ".shp")
        new_shapefile = output_shapefile.CreateLayer(
            output_shp, srs=osr.SpatialReference(wkt=prj)
        )
        gdal.Polygonize(input_band, input_band, new_shapefile, -1, [], callback=None)
        new_shapefile.SyncToDisk()


Polygonized.mask_to_raster(input_raster, output_dir)
