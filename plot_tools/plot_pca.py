import torch
import torch.nn as nn
import argparse
import sys
sys.path.append('.')
import pickle
from model_architectures.UNetModelv4 import UNetModelv4
from mask_training.models import CombinedModel, Identity, MultiLayerModel, NormalizeLayer
from tools.dataloader import Dataset, ShuffleDataset
from tools.pytorch_essentials import predict, get_device
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm


DATASET_CONFIG_PATH = "datasets/cu_mnist.json"
MODEL_FILENAME = "models/model1.pt"


if __name__ == "__main__":
    batch_size = 128
    device = get_device()
    model = torch.load(MODEL_FILENAME, weights_only=False, map_location=device)

    if isinstance(model, MultiLayerModel):
        if isinstance(model.layers[-1], NormalizeLayer):
            model.layers = model.layers[:-1]

    dataset = Dataset(True, DATASET_CONFIG_PATH, mode="classifier")

    dataset = ShuffleDataset(dataset, batch_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=1, pin_memory=True)
    
    import itertools
    max_batches = 8
    loader = itertools.islice(loader, max_batches)    

    spectra = []
    labels = []

    for image, label, description in tqdm(loader, total=max_batches):
        prediction = predict(model, device, image)
        spectra.append(prediction)
        labels.append(label)

    spectra = np.concatenate(spectra, axis=0)
    labels = np.concatenate(labels, axis=0)

    original_shape = spectra.shape[1:]
    mean_spectrum = np.mean(spectra, axis=0)
    spectra = spectra.reshape(spectra.shape[0], -1)

    pca = PCA(n_components=10)

    spectra_pca = pca.fit_transform(spectra)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=spectra_pca[:, 0], y=spectra_pca[:, 1],
        hue=labels,
        palette=sns.color_palette("hls", 10),
        legend="full",
        alpha=0.3,
    )
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")

    plt.title('PCA of Mnist Digits Spectra')



    reshaped_components = np.array([pc.reshape(original_shape) for pc in pca.components_[:10]])

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    axes = axes.flatten()

    vmin, vmax = np.min(reshaped_components), np.max(reshaped_components)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    for i, ax in enumerate(axes):
        im = ax.imshow(reshaped_components[i], cmap='seismic', norm=norm, aspect="auto")
        ax.set_title(f"PCA component {i+1}")
        ax.axis('off')  

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)


    fig, ax = plt.subplots(figsize=(10, 6))

    fig.subplots_adjust(left=0.05, right=0.75)  

    slider_axes = []
    sliders = []
    for i in range(10):
        ax_slider = plt.axes([0.8, 0.75 - i*0.062, 0.15, 0.05])
        slider_axes.append(ax_slider)
        
        slider = Slider(ax_slider, f'PC{i+1}', -3, 3, valinit=0)
        slider.poly.set_facecolor('#ADD8E6')
        slider.vline.set_color('#000000')
        sliders.append(slider)

    norm = mcolors.Normalize(vmin=0, vmax=0.3)
    im = ax.imshow(mean_spectrum, cmap='viridis', interpolation='none', norm=norm)
    ax.set_title('PCA Components Visualization')
    ax.axis('off')

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)

    def update(val):
        new_spectrum = mean_spectrum + sum(sliders[i].val * reshaped_components[i] for i in range(10))
        im.set_data(new_spectrum)
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)

    plt.show()

