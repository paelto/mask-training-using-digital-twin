import os
import sys
sys.path.append('.')
import datasets as ds
import pickle
import numpy as np

from tools.prepare_data import Preparer

SOURCE_PATH = "datasets/20241108_cifar10 and friends/source_images"
INPUT_DATA_PATH = "datasets/custom/input_data"

prep = Preparer(INPUT_DATA_PATH, None, (100,100), 1000)

mnist = ds.load(os.path.join(SOURCE_PATH, "mnist_full_rotated.ds"))
prep.preprocess(
    mnist.raw,
    mnist.label,
    "mnist",
    1,
    13,
    grayscale=True,
    invert=True,
    pad_level=255,
    power=350
)

SOURCE_PATH = "datasets/custom/source_images"
INPUT_DATA_PATH = "datasets/custom/input_data"

cifar_raw = []
cifar_label = []

for i in range(0,6):
    if i == 0:
        name = os.path.join(SOURCE_PATH, "cifar-10-python/cifar-10-batches-py/test_batch")
    else:
        name = os.path.join(SOURCE_PATH, f"cifar-10-python/cifar-10-batches-py/data_batch_{i}")

    with open(name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    cifar_raw.append(dict[b"data"])
    cifar_label.append(dict[b"labels"])

    del dict

cifar_raw = np.concatenate(cifar_raw)
cifar_raw = cifar_raw.reshape(-1, 3, 32, 32)
cifar_raw = np.mean(cifar_raw, axis=1)

cifar_label = np.concatenate(cifar_label)

prep = Preparer(INPUT_DATA_PATH, None, (100,100), 1000)

prep.preprocess(
    cifar_raw,
    cifar_label,
    "cifar",
    0,
    12,
    grayscale=True,
    pad_level=255,
    power=400
)