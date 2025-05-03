import os
import sys
sys.path.append('.')
import datasets as ds

from tools.prepare_data import Preparer

SOURCE_PATH = "datasets/20241101_sanity checks/source_images"
INPUT_DATA_PATH = "datasets/custom/input_data"

prep = Preparer(INPUT_DATA_PATH, None, (100,100), 1000)

fashion = ds.load(os.path.join(SOURCE_PATH, "fashion_70k.ds"))

print(fashion.raw.shape)

prep.preprocess(
    fashion.raw,
    fashion.label,
    "fashion",
    0,
    13,
    grayscale=True,
    invert=True,
    pad_level=255,
    power=350
)