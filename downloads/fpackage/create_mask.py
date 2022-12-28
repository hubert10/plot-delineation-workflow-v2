import numpy as np
from shapely.ops import cascaded_union
import rasterio
import rasterio.mask
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Polygon
from skimage.morphology import square, dilation


class CreateMask:
    def __init__(self, raster_path, shape_path):
        self.raster_path = raster_path
        self.shape_path = shape_path

    def __poly_from_utm(self, polygon, transform):
        poly_pts = []

        poly = cascaded_union(polygon)
        for i in np.array(poly.exterior.coords):

            # Convert polygons to the image CRS
            poly_pts.append(~transform * tuple(i))

        # Generate a polygon object
        new_poly = Polygon(poly_pts)
        return new_poly

    def binarize_raster(self):
        with rasterio.open(self.raster_path, "r") as src:
            raster_img = src.read()
        train_df = gpd.read_file(self.shape_path)

        # Generate Binary maks
        poly_shp = []
        im_size = (src.meta["height"], src.meta["width"])
        for num, row in train_df.iterrows():
            if row["geometry"].geom_type == "Polygon":
                poly = self.__poly_from_utm(row["geometry"], src.meta["transform"])
                poly_shp.append(poly)
            else:
                for p in row["geometry"]:
                    poly = self.__poly_from_utm(p, src.meta["transform"])
                    poly_shp.append(poly)

        mask = rasterize(shapes=poly_shp, out_shape=im_size)
        return mask

    def get_boundaries(self, mask, boundary_width=20):
        footprint_msk = mask * 255
        strel = square(boundary_width)
        boundary_mask = dilation(footprint_msk, strel)
        boundary_mask = boundary_mask ^ footprint_msk
        burn_value = 255
        boundary_mask = boundary_mask > 0  # need to binarize to get burn val right
        output_arr = boundary_mask.astype("uint8") * burn_value

        return output_arr

    def save_mask(self, output_arr, out_file):
        with rasterio.open(self.raster_path, "r") as src:
            raster_img = src.read()
        meta = src.meta.copy()
        meta.update({"count": 1})
        meta.update(dtype="uint8")
        meta.update(nodata=0)
        with rasterio.open(out_file, "w", **meta) as dst:
            dst.write(output_arr, indexes=1)
