import os
import sys
sys.path.append('.')
import datasets as ds

from tools.prepare_data import Preparer

SOURCE_PATH = "datasets/20241101_sanity checks/source_images"
INPUT_DATA_PATH = "datasets/20241101_sanity checks/input_data"
OUTPUT_DATA_PATH = "datasets/20241101_sanity checks/data"

prep = Preparer(INPUT_DATA_PATH, OUTPUT_DATA_PATH, (100,100), 1000)

ham28 = ds.load(os.path.join(SOURCE_PATH, "ham28_shuffled.ds"))
for rot in [0,1]:
    prep.preprocess(
        ham28.raw,
        ham28.label,
        f"ham28_shuffled_gTrue_rot{rot}_",
        rot-1,
        13,
        pad_level=166,
        power = 310
    )

prep.preprocess(
    ham28.raw,
    ham28.label,
    "ham28_shuffled_higherP_gTrue_rot0_",
    0,
    13,
    pad_level=166,
    power = 400
)
del ham28

ham95 = ds.load(os.path.join(SOURCE_PATH, "ham95_shuffled.ds"))
for rot in [0,1]:
    prep.preprocess(
        ham95.raw,
        ham95.label,
        f"ham95_shuffled_gTrue_rot{rot}_",
        rot-1,
        4,
        pad_level=166,
        power = 310
    )
del ham95

md_ham28 = ds.load(os.path.join(SOURCE_PATH, "ham28_shuffled_mode_patterns.ds"))
for mode in range(10):
    prep.preprocess(
        (md_ham28.take(mode=mode).raw > 0)*255,
        md_ham28.label,
        f"ham28_shuffled_mode{mode}_gFalse_rot0_",
        -1,
        14,
        grayscale=False,
        invert=True,
        power = 310
    )
del md_ham28

md_ham95 = ds.load(os.path.join(SOURCE_PATH, "ham95_shuffled_mode_patterns.ds"))
for mode in range(10):
    prep.preprocess(
        (md_ham95.take(mode=mode).raw > 0)*255,
        md_ham95.label,
        f"ham95_shuffled_mode{mode}_gFalse_rot0_",
        -1,
        4,
        grayscale=False,
        invert=True,
        power = 310
    )
del md_ham95

ham28_k = ds.load(os.path.join(SOURCE_PATH, "ham28_kaggle_shuffled.ds"))
prep.preprocess(
    ham28_k.raw,
    ham28_k.label,
    "ham28_kaggle_shuffled_gTrue_rot0_",
    -1,
    13,
    pad_level=166,
    power = 310
)
del ham28_k

ham95_k = ds.load(os.path.join(SOURCE_PATH, "ham95_kaggle_shuffled.ds"))
prep.preprocess(
    ham95_k.raw,
    ham95_k.label,
    "ham95_kaggle_shuffled_gTrue_rot0_",
    -1,
    4,
    pad_level=166,
    power = 310
)
del ham95_k

ham28_c = ds.load(os.path.join(SOURCE_PATH, "ham28_contrast.ds"))
prep.preprocess(
    ham28_c.raw,
    ham28_c.label,
    "ham28_contrast_gTrue_rot0_",
    -1,
    13,
    pad_level=230,
    power = 310
)
del ham28_c

bh = ds.load(os.path.join(SOURCE_PATH, "breakhist_div2_unquart.ds"))
for rot in [0,1]:
    prep.preprocess(
        bh.raw,
        bh.label,
        f"breakhist_div2_unquart_gTrue_rot{rot}_",
        rot-1,
        1,
        pad_level=213,
        power = 310
    )
del bh

isic = ds.load(os.path.join(SOURCE_PATH, "isic12_95.ds"))
for rot in [0,1]:
    prep.preprocess(
        isic.raw,
        isic.label,
        f"isic12_95_gTrue_rot{rot}_",
        rot,
        4,
        pad_level=157,
        power = 310
    )
del isic

md_isic = ds.load(os.path.join(SOURCE_PATH, "isic12_95_mode_patterns.ds"))
for mode in range(10):
    prep.preprocess(
        (md_isic.take(mode=mode).raw > 0)*255,
        md_isic.label,
        f"isic95_mode{mode}_gFalse_rot0_",
        0,
        4,
        grayscale=False,
        invert=True,
        power = 310
    )
del md_isic

isic = ds.load(os.path.join(SOURCE_PATH, "isic12_190.ds"))
for rot in [0,1]:
    prep.preprocess(
        isic.raw,
        isic.label,
        f"isic12_190_gTrue_rot{rot}_",
        rot,
        2,
        pad_level=159,
        power = 310
    )
del isic

md_isic = ds.load(os.path.join(SOURCE_PATH, "isic12_190_mode_patterns.ds"))
for mode in range(10):
    prep.preprocess(
        (md_isic.take(mode=mode).raw > 0)*255,
        md_isic.label,
        f"isic190_mode{mode}_gFalse_rot0_",
        0,
        2,
        grayscale=False,
        invert=True,
        power = 310
    )
del md_isic


c10g = ds.load(os.path.join(SOURCE_PATH, "cifar10_gray.ds"))
prep.preprocess(
    c10g.raw,
    c10g.label,
    "cifar10_gray_gTrue_rot0_",
    0,
    12,
    pad_level=255,
    power = 400
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
        pad_level=255,
        power = 400
    )
del c10

md_cifar = ds.load(os.path.join(SOURCE_PATH, "cifar10_gray_mode_patterns.ds"))
for mode in range(2, 4):
    prep.preprocess(
        (md_cifar.take(mode=mode).raw > 0)*255,
        md_cifar.label,
        f"cifar10_mode{mode}_gFalse_rot0_",
        0,
        13,
        grayscale=False,
        invert=True,
        power = 350
    )
del md_cifar

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
    power=310)

prep.preprocess(
    mnist.raw[:40000],
    mnist.label[:40000],
    "mnist_rot0_late_",
    1,
    19,
    grayscale=False,
    invert=True,
    pad_level=255,
    power=350)
del mnist

"""
positioner
taskB_K11S3_
taskB_K4S3_
taskB_K7S3_
taskG_K11S3_
taskG_K4S3_
taskG_K7S3_
taskR_K11S3_
taskR_K4S3_
taskR_K7S3_
tube2I_K11S3_
tube2I_K4S3_
tube2I_K7S3_
"""