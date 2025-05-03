import torch
import matplotlib.pyplot as plt
import argparse
import sys
import torch.nn.functional as F
sys.path.append('.')

from tools.pytorch_essentials import get_device
from mask_training.models import MultiLayerModel, NormalizeLayer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for plotting masks")
    parser.add_argument('model_filename', type=str, help="Filename of trained model")

    args = parser.parse_args()

    device = get_device()
    print(f"Running on {device}")

    model = torch.load(args.model_filename, weights_only=False, map_location=device)
    model.to(device)

    if isinstance(model, MultiLayerModel):
        if isinstance(model.layers[-1], NormalizeLayer):
            model.layers = model.layers[:-1]

    if hasattr(model.layers[-1], 'linear_layer'):
        weights = model.layers[-1].linear_layer.weight
        bn = model.layers[-1].bn
        weights = weights * bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weights = weights.unflatten(1, (64, 128)).detach()

    if hasattr(model.layers[0], 'a'):
        fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    else:
        fig, axs = plt.subplots(3, 4, figsize=(15, 15))

    if hasattr(model.layers[-1], 'linear_layer'):
        for i in range(10):
            axs[i//4, i%4].imshow(weights[i])
            axs[i//4, i%4].set_title(f"weights {i}")

        axs[2, 2].imshow(bn.running_mean.unflatten(0, (64, 128)).detach())
        axs[2, 2].set_title("mean")
        axs[2, 3].imshow(bn.running_var.unflatten(0, (64, 128)).detach())
        axs[2, 3].set_title("variance")

    if hasattr(model.layers[0], 'a'):
        a_padded, b_padded = model.layers[0].get_scaled_mask()
        a_padded = a_padded.detach()
        b_padded = b_padded.detach()
        axs[3, 0].imshow(a_padded)
        axs[3, 0].set_title("a scaled")
        axs[3, 1].imshow(b_padded)
        axs[3, 1].set_title("b scaled")
        axs[3, 2].set_visible(False)
        axs[3, 3].set_visible(False)


    plt.show()
