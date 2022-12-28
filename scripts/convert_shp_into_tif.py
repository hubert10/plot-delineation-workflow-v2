"""
@author: Steven Ndung'u' and Hubert K

"""
# ## Georeference Masks
# Example found in the workstation workflow
# The aim of this script is to convert prediction patches saved in jp, png, etc to TIF format referencing the metadata of the original satelite imagery

import os
import rasterio as rio
import skimage.io as io
from datetime import datetime
import pathlib
from utils.make_dir import create_dir
from utils.config import PROJECT_ROOT
from utils.config import roi_image
from osgeo import gdal
import ogr, gdal, osr
from flusstools import geotools

gdal.UseExceptions()

#### Convert the PNG predictions to Rasters Tif format

# path of png or jpg image predicted from smoothing algorithm
# input_img_jpg = PROJECT_ROOT + "results/Test/predicted/"
input_img_jpg = PROJECT_ROOT + "results/Test/filtered/"
input_img_jpg = input_img_jpg + roi_image.split(".")[0]
input_img_jpg = pathlib.Path(input_img_jpg)

# folder with original raster image (from original tif)
georef_img_tif = PROJECT_ROOT + "results/Test/inputs/"
georef_img_tif = pathlib.Path(georef_img_tif)

# Path to save the outputs
save_path = PROJECT_ROOT + "results/Test/georeferenced/"
# Create dir for saving predictions
output_dir = create_dir(save_path + roi_image.split(".")[0])
save_path = pathlib.Path(output_dir)
input_img_jpg, georef_img_tif, save_path


in_shp_file_name = "/home/hubert/Desktop/Heuristics/RS/plot-delineation-workflow/samples/roi/MaliDec2016/Couche_gamma.shp"
out_raster_file_name = "/home/hubert/Desktop/Heuristics/RS/plot-delineation-workflow/samples/roi/DebiTiguet/debi-tiguet_image.shp"
# out_raster_file_name="/home/hubert/Desktop/Heuristics/plot-delineation-workflow/samples/roi/DebiTiguet/debi-tiguet_image.tif"


def rasterize(
    in_shp_file_name,
    out_raster_file_name,
    pixel_size=10,
    no_data_value=-9999,
    rdtype=gdal.GDT_Float32,
    **kwargs
):
    """
    Converts any shapefile to a raster
    :param in_shp_file_name: STR of a shapefile name (with directory e.g., "C:/temp/poly.shp")
    :param out_raster_file_name: STR of target file name, including directory; must end on ".tif"
    :param pixel_size: INT of pixel size (default: 10)
    :param no_data_value: Numeric (INT/FLOAT) for no-data pixels (default: -9999)
    :param rdtype: gdal.GDALDataType raster data type - default=gdal.GDT_Float32 (32 bit floating point)
    :kwarg field_name: name of the shapefile's field with values to burn to the raster
    :return: produces the shapefile defined with in_shp_file_name
    """

    # open data source
    try:
        source_ds = ogr.Open(in_shp_file_name)
    except RuntimeError as e:
        print("Error: Could not open %s." % str(in_shp_file_name))
        return None
    source_lyr = source_ds.GetLayer()

    # read extent
    x_min, x_max, y_min, y_max = source_lyr.GetExtent()
    print(source_lyr.GetExtent())
    # get x and y resolution
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    # create destination data source (GeoTIff raster)
    target_ds = gdal.GetDriverByName("GTiff").Create(
        out_raster_file_name, x_res, y_res, 1, eType=rdtype
    )
    # target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, 100, 100, 1, gdal.GDT_Byte)

    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    band = target_ds.GetRasterBand(1)
    band.Fill(no_data_value)
    band.SetNoDataValue(no_data_value)

    # get spatial reference system and assign to raster
    srs = geotools.get_srs(source_ds)
    try:
        srs.ImportFromEPSG(int(srs.GetAuthorityCode(None)))
    except RuntimeError as e:
        print(e)
        return None
    target_ds.SetProjection(srs.ExportToWkt())

    # RasterizeLayer(Dataset dataset, int bands, Layer layer, pfnTransformer=None, pTransformArg=None,
    # int burn_values=0, options=None, GDALProgressFunc callback=0, callback_data=None)
    gdal.RasterizeLayer(
        target_ds,
        [1],
        source_lyr,
        None,
        None,
        burn_values=[0],
        options=["ALL_TOUCHED=TRUE", "ATTRIBUTE=" + str(kwargs.get("field_name"))],
    )

    # release raster band
    band.FlushCache()


rasterize(in_shp_file_name, out_raster_file_name)


# def Feature_to_Raster(input_shp, output_tiff, cellsize, field_name=False, NoData_value=-9999):
#     """
#     Converts a shapefile into a raster
#     https://www.programcreek.com/python/?code=wateraccounting%2Fwa%2Fwa-master%2FModels%2Fwaterpix%2Fwp_gdal%2Fdavgis%2Ffunctions.py
#     """

#     now = datetime.now()  # current date and time
#     time = now.strftime("%m%d%Y_%H%M")

#     # Input
#     inp_driver = ogr.GetDriverByName('ESRI Shapefile')
#     inp_source = inp_driver.Open(input_shp, 0)
#     inp_lyr = inp_source.GetLayer()
#     inp_srs = inp_lyr.GetSpatialRef()

#     # Extent
#     x_min, x_max, y_min, y_max = inp_lyr.GetExtent()
#     print(inp_lyr.GetExtent())
#     x_ncells = int((x_max - x_min) / cellsize)
#     y_ncells = int((y_max - y_min) / cellsize)
#     print(x_ncells)
#     print(y_ncells)
#     # Output
#     out_driver = gdal.GetDriverByName('GTiff')
#     if os.path.exists(output_tiff):
#         out_driver.Delete(output_tiff)
#     out_source = out_driver.Create(output_tiff, x_ncells, y_ncells,1, gdal.GDT_Int16)
#     print("+"*50)
#     print(x_ncells, y_ncells,1, gdal.GDT_Int16)

#     out_source.SetGeoTransform((x_min, cellsize, 0, y_max, 0, -cellsize))
#     out_source.SetProjection(inp_srs.ExportToWkt())
#     out_lyr = out_source.GetRasterBand(1)
#     out_lyr.SetNoDataValue(NoData_value)

#     # Rasterize
#     # print(inp_lyr)
#     if field_name:
#         gdal.RasterizeLayer(out_source, [1], inp_lyr,options=["ATTRIBUTE={0}".format(field_name)])
#     else:
#         gdal.RasterizeLayer(out_source, [1], inp_lyr, burn_values=[1])

#     # Save and/or close the data sources
#     inp_source = None
#     out_source = None
#     print(output_tiff)

# #     io.imsave(
# #     os.path.join(
# #         PROJECT_ROOT + "results/Test/geo_referenced",
# #         "RGB_from_{}{}{}".format(
# #             str(PROJECT_ROOT).split("/")[-1].split("/")[0], str(time), ".tif"
# #         ),
# #     ),
# #     output_tiff,
# # )
#     # Return
#     print(output_tiff)
#     return output_tiff

# #geo_silhouettes
# Feature_to_Raster(in_shp_file_name, out_raster_file_name, 0.001)

# output_raster = out_raster_file_name


# def main(shapefile):

#     # making the shapefile as an object.
#     input_shp = ogr.Open(shapefile)

#     # getting layer information of shapefile.
#     shp_layer = input_shp.GetLayer()

#     # pixel_size determines the size of the new raster.
#     # pixel_size is proportional to size of shapefile.
#     pixel_size = 0.1

#     # get extent values to set size of output raster.
#     x_min, x_max, y_min, y_max = shp_layer.GetExtent()
#     print(shp_layer.GetExtent())

#     # calculate size/resolution of the raster.
#     x_res = int((x_max - x_min) / pixel_size)
#     y_res = int((y_max - y_min) / pixel_size)
#     # get GeoTiff driver by
#     image_type = "GTiff"
#     driver = gdal.GetDriverByName(image_type)

#     # passing the filename, x and y direction resolution, no. of bands, new raster.
#     new_raster = driver.Create(output_raster, x_res, y_res, 1, gdal.GDT_Byte)

#     # transforms between pixel raster space to projection coordinate space.
#     new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))

#     # get required raster band.
#     band = new_raster.GetRasterBand(1)

#     # assign no data value to empty cells.
#     no_data_value = -9999
#     band.SetNoDataValue(no_data_value)
#     band.FlushCache()

#     # main conversion method
#     gdal.RasterizeLayer(new_raster, [1], shp_layer, burn_values=[255])

#     # adding a spatial reference
#     new_rasterSRS = osr.SpatialReference()
#     new_rasterSRS.ImportFromEPSG(2975)
#     new_raster.SetProjection(new_rasterSRS.ExportToWkt())
#     return gdal.Open(output_raster)


# main(in_shp_file_name)
