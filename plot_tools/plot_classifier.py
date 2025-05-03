import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import inspect
import sys
sys.path.append('.')

from tools.dataloader import Dataset, ShuffleDataset
from tools.pytorch_essentials import get_device

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

    test_dataset = Dataset(False, args.dataset_config_path, mode="classifier_spectrum" if model.input_shape==(64,128) else "classifier")
    test_dataset = ShuffleDataset(test_dataset, args.buffer_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)
    
    fig = plt.figure(figsize=(15, 15))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    gs = gridspec.GridSpec(args.n_rows+1, len(model.layers) + 1, figure=fig, wspace=0.4, hspace=0.6, height_ratios=[1]*args.n_rows + [0.1])

#    titles = ["input", "1st spectrum", "PCA grid", "masked PCA grid", "2nd spectrum", "prediction"]
    titles = [str(x) for x in range(len(model.layers) + 1)]

    image, category, description = next(iter(test_loader))

    if "cifar" in description[0]:
        categories = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    elif "fashion" in description[0]:
        categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        categories = range(10)        

    model.eval()

    input_image = image = image[:args.n_rows]

    for i, layer in enumerate([*model.layers, None]):
        for x in range(args.n_rows):
            ax = fig.add_subplot(gs[x, i])

            if x == 0:
                ax.set_title(titles[i], fontsize=10)
            if i == 0:
                ax.text(-0.6, 0.5, description[x], ha='center', va='center', fontsize=10, rotation=90, transform=ax.transAxes)


            if len(image.shape) == 3:
                if image[x].shape == (64,128):
                    ax.set_xlabel("$\\lambda$ [nm]")
                    extend = [816.21, 942.24, 64, 0]
                else:
                    ax.set_xlabel("x [px]")
                    extend = None

                im = ax.imshow(image[x].detach().cpu().numpy(), vmin=0, extent=extend)
                ax.set_ylabel("y [px]")

            elif len(image.shape) == 2:
                prediction = torch.softmax(image, dim=1).detach().cpu().numpy()
                ax.bar(categories, prediction[x], color='lightblue')
                ax.set_xticks(categories)
                if len(str(categories[0])) > 5:
                    ax.set_xticklabels(categories, rotation=60, ha='right', va='top', rotation_mode='anchor')
                else:
                    ax.set_xlabel("Category")
                ax.get_xticklabels()[category[x]].set_fontweight('bold')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")

        if len(image.shape) == 3:
            ax = fig.add_subplot(gs[args.n_rows, i])
            fig.colorbar(im, cax=ax, orientation='horizontal', pad=0.1)

        if layer != None:
            if 'input_image' in inspect.signature(layer.forward).parameters:
                image = layer(image, input_image=input_image)
            else:
                image = layer(image)
    

    plt.show()