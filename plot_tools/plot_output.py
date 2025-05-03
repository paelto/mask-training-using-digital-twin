import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import sys
sys.path.append('.')

from tools.dataloader import Dataset, ShuffleDataset
from tools.pytorch_essentials import predict, get_device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for plotting hyperspectra")
    parser.add_argument('dataset_config_path', type=str, help="Path to the dataset configuration file")
    parser.add_argument('model_filename', type=str, help="Filename of trained model")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument('--buffer_size', type=int, default=5000, help="Buffer size (default: 5000)")
    parser.add_argument('--n_rows', type=int, default=6, help="Number of rows to plot (default: 6)")
    parser.add_argument('--mode', type=str, default="forward", help="Mode of digital twin", choices=["forward", "reversed"])

    args = parser.parse_args()

    device = get_device()
    print(f"Running on {device}")

    model = torch.load(args.model_filename, weights_only=False, map_location=device)
    model.to(device)

    test_dataset = Dataset(False, args.dataset_config_path, mode=args.mode)
    test_dataset = ShuffleDataset(test_dataset, args.buffer_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.8)

    gs = gridspec.GridSpec(args.n_rows+1, 3, figure=fig, width_ratios=[1.5, 2, 2], height_ratios=[1]*args.n_rows + [0.1])

    image_in, image_out, description = next(iter(test_loader))

    image_in = image_in[:args.n_rows]
    image_out = image_out[:args.n_rows]
    description = description[:args.n_rows]

    prediction = predict(model, device, image_in)

    real_flattend = image_out.reshape(len(image_out), -1)
    pred_flattend = prediction.reshape(len(prediction), -1)

    for x in range(args.n_rows):
        ax = fig.add_subplot(gs[x, 0])
        im1 = ax.imshow(image_in[x], vmin=0, vmax=image_in.max())
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        if x == 0:
            ax.set_title("input", fontsize=10)

        ax = fig.add_subplot(gs[x, 1])
        im2 = ax.imshow(image_out[x], vmin=0, vmax=max(image_out.max(), prediction.max()), extent=[816.21, 942.24, 64, 0])
        ax.set_xlabel("$\\lambda$ [nm]")
        ax.set_ylabel("y [px]")
        if x == 0:
            ax.set_title("real output", fontsize=10)

        ax = fig.add_subplot(gs[x, 2])
        ax.imshow(prediction[x], vmin=0, vmax=max(image_out.max(), prediction.max()), extent=[816.21, 942.24, 64, 0])
        ax.set_xlabel("$\\lambda$ [nm]")
        ax.set_ylabel("y [px]")
        if x == 0:
            ax.set_title("predicted output", fontsize=10)

    cax = fig.add_subplot(gs[args.n_rows, 0])
    fig.colorbar(im1, cax=cax, orientation='horizontal', pad=0.1)

    cax = fig.add_subplot(gs[args.n_rows, 1:3])
    fig.colorbar(im2, cax=cax, orientation='horizontal', pad=0.1)



    plt.show()