import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize
import torch
import numpy as np
import tqdm as tqdm
import sys
sys.path.append('.')

from tools.dataloader import Dataset, ShuffleDataset
from tools.pytorch_essentials import get_device, predict
from mask_training.models import NormalizeLayer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for plotting hyperspectra")
    parser.add_argument('dataset_config_path', type=str, help="Path to the dataset configuration file")
    parser.add_argument('model_filename', type=str, help="Filename of trained model")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument('--buffer_size', type=int, default=5000, help="Buffer size (default: 5000)")
    parser.add_argument('--n_rows', type=int, default=6, help="Number of rows to plot (default: 6)")

    args = parser.parse_args()

    device = get_device()
    print(f"Running on {device}")

    model = torch.load(args.model_filename, weights_only=False, map_location=device)
    model.to(device)

    if hasattr(model, "layers"):
        if isinstance(model.layers[-1], NormalizeLayer) or len(model.layers[-1].output_shape) == 1:
            model.layers = model.layers[:-1]


    test_dataset = Dataset(False, args.dataset_config_path, mode="classifier_spectrum" if model.input_shape==(64,128) else "classifier")
    test_dataset = ShuffleDataset(test_dataset, args.buffer_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)


    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(args.n_rows+6, 11, figure=fig, width_ratios=[1]*10 + [0.15], height_ratios=[0.2, 0.2, 1, 0.2, 1, 0.2] + [1]*args.n_rows)

    spectra = [[] for _ in range(10)]

    for images, labels, descriptions in test_loader:
        predictions = predict(model, device, images)
        for prediction, label in zip(predictions, labels):
            spectra[label].append(prediction)
        if min([len(x) for x in spectra]) >= args.n_rows:
            break

    spectra = [x[:args.n_rows] for x in spectra]
    spectra = np.array(spectra)

    mean = np.mean(spectra, axis=1)
    std = np.std(spectra, axis=1)

    total_mean = np.mean(spectra, axis=(0, 1))

    spectra_diff = spectra - total_mean
    mean_diff = mean - total_mean

    norm_spectra = TwoSlopeNorm(vmin=np.min(spectra_diff), vcenter=0, vmax=np.max(spectra_diff))
    norm_std = Normalize(vmin=0, vmax=np.max(std))

    if "cifar" in descriptions[0]:
        categories = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    elif "fashion" in descriptions[0]:
        categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        categories = range(10)

    ax = fig.add_subplot(gs[1, :])
    ax.text(0.5, 0.5, "standard deviation", ha='center', va='center', fontsize=12)
    ax.axis('off')

    ax = fig.add_subplot(gs[3, :])
    ax.text(0.5, 0.5, "mean spectra (difference to total mean)", ha='center', va='center', fontsize=12)
    ax.axis('off')

    ax = fig.add_subplot(gs[5, :])
    ax.text(0.5, 0.5, "random spectra (difference to total mean)", ha='center', va='center', fontsize=12)
    ax.axis('off')

    for i, category in enumerate(categories):
        ax = fig.add_subplot(gs[0, i])
        ax.text(0.5, 0.5, category, ha='center', va='center', fontsize=12)
        ax.axis('off')

        ax = fig.add_subplot(gs[2, i])
        ax.set_xticks([])
        ax.set_yticks([])
        im_std = ax.imshow(std[i], norm=norm_std, cmap='Oranges')

        ax = fig.add_subplot(gs[4, i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(mean_diff[i], norm=norm_spectra, cmap='seismic')

        for x in range(args.n_rows):
            ax = fig.add_subplot(gs[x+6, i])
            ax.set_xticks([])
            ax.set_yticks([])
            im_spectra = ax.imshow(spectra_diff[i][x], norm=norm_spectra, cmap='seismic')

    ax = fig.add_subplot(gs[2, -1])
    fig.colorbar(im_std, cax=ax, orientation="vertical")

    ax = fig.add_subplot(gs[4:, -1])
    fig.colorbar(im_spectra, cax=ax, orientation="vertical")


    plt.tight_layout(w_pad=0.5, h_pad=0.5)
    plt.show()
