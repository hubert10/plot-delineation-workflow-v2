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
import matplotlib.pyplot as plt
from utils.config import PROJECT_ROOT
import geopandas as gpd
from utils.config import roi_image
import sys, os
sys.path.append("./")
import pathlib

# Claculating the lenth and area of polygons
# Link: https://gis.stackexchange.com/questions/287069/calculating-length-of-polygon-in-geopandas
# https://stackoverflow.com/questions/35878064/plot-two-histograms-on-the-same-graph-and-have-their-columns-sum-to-100
# Load original Image

roi_file_path = (
    PROJECT_ROOT + "samples/roi/DebiTiguet/debi_tiguet.geojson"
)

# Load the predicted geojsons plots
pred_file_path = PROJECT_ROOT + "results/savedfiles/"
pred_file_path = pred_file_path + roi_image.split(".")[0] 
pred_file_path = pathlib.Path(pred_file_path)

original_df = gpd.read_file(roi_file_path)
pred_df = gpd.read_file(os.path.join(pred_file_path, roi_image.split(".")[0] + '.geojson'))

ECKERT_IV_PROJ4_STRING = (
    "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
original_df = original_df.to_crs(ECKERT_IV_PROJ4_STRING)
original_df["area"] = original_df["geometry"].area
# original_df.head()

pred_df = pred_df.to_crs(ECKERT_IV_PROJ4_STRING)
pred_df["area"] = pred_df["geometry"].area
# pred_df.head()
pred_df = pred_df[pred_df.area > 50]
pred_df = pred_df[pred_df.area < 20420]

print(original_df.shape)
print(pred_df.shape)
# you can round the result to 2 digit by using a lambda function
# gdf['rounded_area'] = gdf['area'].apply(lambda x: round(x, 2))
fig, ax = plt.subplots()
ax.hist(original_df["area"], color="lightblue", label="ground truth plots", alpha=0.5)
ax.hist(pred_df["area"], color="salmon", label="predicted plots", alpha=0.5)
plt.legend()

ax.set(
    title="Plots Size Distribution",
    xlabel="Area in Sqm Units",
    ylabel="# of Plots in Bin",
)
ax.margins(0.05)
ax.set_ylim(bottom=0)
plt.show()


# See the comments below for the next step of this prediction
#########################################################################################
#              geospatial raster and vector formats
#              run: python visualization/5_plot_size_distribution.py
#########################################################################################
