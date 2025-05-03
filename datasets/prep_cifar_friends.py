import os
import sys
sys.path.append('.')
import datasets as ds

from tools.prepare_data import Preparer

SOURCE_PATH = "datasets/20241108_cifar10 and friends/source_images"
INPUT_DATA_PATH = "datasets/20241108_cifar10 and friends/input_data"
OUTPUT_DATA_PATH = "datasets/20241108_cifar10 and friends/data"

prep = Preparer(INPUT_DATA_PATH, OUTPUT_DATA_PATH, (100,100), 1000)

ham28 = ds.load(os.path.join(SOURCE_PATH, "ham28_shuffled.ds"))
prep.preprocess(
    ham28.raw,
    ham28.label,
    "ham28_shuffled_gTrue_rot0_",
    0,
    13,
    grayscale=True,
    invert=False,
    pad_level=166,
    power=400
)
del ham28

ham95 = ds.load(os.path.join(SOURCE_PATH, "ham95_shuffled.ds"))
prep.preprocess(
    ham95.raw,
    ham95.label,
    "ham95_shuffled_gTrue_rot0_",
    0,
    4,
    grayscale=True,
    invert=False,
    pad_level=166,
    power=400
)
del ham95

c10g = ds.load(os.path.join(SOURCE_PATH, "cifar10_gray.ds"))
prep.preprocess(
    c10g.raw,
    c10g.label,
    "cifar10_gray_gTrue_rot0_",
    0,
    12,
    grayscale=True,
    invert=False,
    pad_level=255,
    power=400
)
del c10g

c10 = ds.load(os.path.join(SOURCE_PATH, "cifar10.ds"))
for i, c in enumerate("rgb"):
    prep.preprocess(
        c10.raw[:, :, :, i],
        c10.label,
        f"cifar10_{c}_gTrue_rot0_",
        0,
        12,
        grayscale=True,
        invert=False,
        pad_level=255,
        power=400
    )
del c10

md_cifar = ds.load(os.path.join(SOURCE_PATH, "cifar10_gray_mode_patterns.ds"))
for mode in range(10):
    prep.preprocess(
        (md_cifar.take(mode=mode).raw > 0)*255,
        md_cifar.label,
        f"cifar10_mode{mode}_gFalse_rot0_",
        0,
        13,
        grayscale=False,
        invert=True,
        pad_level=255,
        power=350,
    )
del md_cifar

isic2020 = ds.load(os.path.join(SOURCE_PATH, "isic2020_95.ds"))
prep.preprocess(
    isic2020.raw,
    isic2020.label,
    "isic2020_gTrue_rot0_",
    0,
    4,
    grayscale=False,
    invert=True,
    pad_level=209,
    power=400
)
del isic2020

mnist = ds.load(os.path.join(SOURCE_PATH, "mnist_full_rotated.ds"))
prep.preprocess(
    mnist.raw[:40000],
    mnist.label[:40000],
    "mnist_rot0_",
    1,
    19,
    grayscale=False,
    invert=True,
    pad_level=255,
    power=350
)
del mnist