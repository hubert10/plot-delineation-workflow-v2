import rasterio as rio
import os
from itertools import product
from rasterio import windows


class SplitData:
    def __init__(
        self,
        in_path,
        input_filename,
        out_path,
        output_filename,
        tiles_width=256,
        tiles_height=256,
    ):
        self.in_path = in_path
        self.input_filename = input_filename
        self.out_path = out_path
        self.output_filename = output_filename
        self.tiles_width = tiles_width
        self.tiles_height = tiles_height

    def __get_tiles(self, ds):
        nols, nrows = ds.meta["width"], ds.meta["height"]
        offsets = product(
            range(0, nols, self.tiles_width), range(0, nrows, self.tiles_height)
        )
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in offsets:
            window = windows.Window(
                col_off=col_off,
                row_off=row_off,
                width=self.tiles_width,
                height=self.tiles_height,
            ).intersection(big_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform

    def split(self):
        with rio.open(os.path.join(self.in_path, self.input_filename)) as inds:
            tile_width, tile_height = self.tiles_width, self.tiles_height

            meta = inds.meta.copy()

            for window, transform in self.__get_tiles(inds):
                print(window)
                meta["transform"] = transform
                meta["width"], meta["height"] = window.width, window.height
                outpath = os.path.join(
                    self.out_path,
                    self.output_filename.format(
                        int(window.col_off), int(window.row_off)
                    ),
                )
                with rio.open(outpath, "w", **meta) as outds:
                    outds.write(inds.read(window=window))
