from pytorch_essentials import predict, get_device

import cv2
import numpy as np
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for plotting hyperspectra and reconstructed image")
    parser.add_argument('image_filename', type=str, help="Path to input image")
    parser.add_argument('model_filename', type=str, help="Filename of trained model")
    parser.add_argument('model_filename_revered', type=str, help="Filename of trained reversed model")

    args = parser.parse_args()

    device = get_device()
    print(f"Running on {device}")

    model = torch.load(args.model_filename, weights_only=False, map_location=device)
    if not isinstance(model, nn.Module):
        from model_architectures.UNetModelv3 import UNetModelv3
        model = UNetModelv3(6, 60, 250, 60, 30, True)
        model.load_state_dict(torch.load(args.model_filename, weights_only=True, map_location=device))
    model.to(device)

    model_revered = torch.load(args.model_filename_revered, weights_only=False, map_location=device)
    model_revered.to(device)

    image = cv2.imread(args.image_filename, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)/255
    image = np.expand_dims(image, axis=0)

    spectrum = predict(model, device, image)
    reconstruced_image = predict(model_revered, device, spectrum)

    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    axs[0].set_title("Input", fontsize=12)
    axs[1].set_title("Spectrum", fontsize=12)
    axs[2].set_title("Reconstructed image", fontsize=12)

    axs[0].imshow(image[0])
    axs[1].imshow(spectrum[0])
    axs[2].imshow(reconstruced_image[0])

    plt.show()
