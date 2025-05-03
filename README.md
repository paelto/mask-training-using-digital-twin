# Introducing trainable weights to a PNN

Build a digital twin of the network laser described in the paper **"Retinomorphic Machine Vision in a Network Laser"**(https://doi.org/10.48550/arXiv.2407.15558) and train input masks for it.

## Overview

This repository provides a PyTorch based framework for simulating a digital twin of a photonic network laser and code to train input masks to increase image classification accuracy for datasets like MNIST and CIFAR-10

## Repository Structure

- `datasets/` — dataset preparation scripts and dataset specifications  
- `mask_training/` — code for training input masks on the digital twin
- `model_architectures/` — digital twin model definitions  
- `plot_tools/` — scripts for result visualization
- `tools/` — helper functions and utilities  
- `train_twin/` — scripts to train the digital twin model

## Dependencies

Dependencies are listed in `requirements.txt`.

This repo also requires `datasets.py` to be in the python path:
https://github.com/jdranczewski/dataset-suite/blob/main/datasets.py  

## Data

The following training data was used, which was recorded on the real photonic system by Jakub Dranczewski:
- 20241108_cifar10 and friends  
- 20241101_sanity checks

## Results

| Configuration | MNIST  | CIFAR-10 |
|---------------|--------|----------|
| No mask       | 95.75% | 34.10%   |
| Single-mask   | 96.58% | 41.23%   |
| Multi-mask    | 98.24% | 49.06%   |

## Usage

### To train the digital twin:

python3 train_twin/train.py datasets/sc_all.json

### To train the accuracy-driven classifier:

python3 mask_training/train_classifier.py datasets/cu_mnist.json {spectrum,twin,maskv1,maskv2} models/model1.pt

### To train the physics-directed classifier:

python3 mask_training/train_mask_pdt.py datasets/cu_mnist.json mask_pdt models/model1.pt

### To monitor training progress use:

tensorboard --logdir logs

optuna-dashboard sqlite:///logs/optuna.sqlite3


