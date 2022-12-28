# Downloads imagery online and saves locally.

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = BASE_DIR + "/"
os.chdir(PROJECT_ROOT)
print(PROJECT_ROOT)
from fpackage.download_data import create_directories, split, download_satellite_image

data_dir = ""

# create the needed directories
create_directories(data_dir)

# api key at some point
api_key = "AIzaSyCQ47Kk5A8LD0odsoBdBTJ5HjuUp8FnV2k"  # You will need to get your own Google Maps Static Images

filename = PROJECT_ROOT + "Input/mali.geojson"
save_path = PROJECT_ROOT + "Data/grid/"
split(filename, save_path)

out_path = PROJECT_ROOT + "Data/image/"
download_satellite_image(out_path, save_path, api_key, filename)
