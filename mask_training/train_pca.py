import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for mask training")
    parser.add_argument('dataset_config_path', type=str, help="Path to the dataset configuration file")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument('--buffer_size', type=int, default=5000, help="Buffer size (default: 5000)")
    parser.add_argument('--n_epochs', type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument('--prune_after', type=int, default=0, help="Prune after n epochs (default: 0)")

    subparsers = parser.add_subparsers(dest="mode", required=True)
    parsers = []
    for mode in ["pcav1"]:
        subparser = subparsers.add_parser(mode)
        subparser.add_argument('model_filename', type=str, help="Filename of base model")
        parsers.append(subparser)

    args = parser.parse_args()

import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from torch.utils.tensorboard import SummaryWriter
import optuna
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import sys
sys.path.append('.')

from tools.dataloader import Dataset, ShuffleDataset
from tools.pytorch_essentials import train_network, get_device, predict
from mask_training.models import Maskv2, MultiLayerModel, BNLinearLayer
from mask_training.models_pca import PCAv1
from mask_training.losses import MaskLoss

LOGS_PATH = "logs"
MODELS_PATH = "mask_training/models"

def objective(trial, args):
#    lr = trial.suggest_float('lr', 3e-4, 5e-3, log=True)
#    gamma = trial.suggest_float('gamma', 0.7, 0.99)
#    lambda_w = trial.suggest_float('lambda_w', 1e-7, 1e-4, log=True)

#    lr_mask = trial.suggest_float('lr_mask', 5e-4, 1e-1, log=True)
#    lambda_a = trial.suggest_float('lambda_a', 1e-7, 1e-2, log=True)

    n_rows = trial.suggest_int('n_rows', 3, 8)
    n_columns = n_rows

#    normalize = trial.suggest_categorical("normalize", [0])

    lr=2.96e-03
    gamma=8.68e-01
    lambda_w=2.78e-05
    lr_mask=8.29e-02
    lambda_a=3.79e-07
    normalize=0

    m = trial.suggest_float('m', 0.5, 1.5)
    c = trial.suggest_float('c', 0, 1)

    model_name = "mnist" if "mnist" in args.dataset_config_path else "cifar" if "cifar" in args.dataset_config_path else "fashion" if "fashion" in args.dataset_config_path else "unknown"

    model_name = f"{model_name}_{args.mode}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model_name = f"{model_name}_lr={lr:.2e}_gamma={gamma:.2e}_lambda_w={lambda_w:.2e}"
    model_name = f"{model_name}_lr_mask={lr_mask:.2e}_lambda_a={lambda_a:.2e}"
    model_name = f"{model_name}_n_rows={n_rows}_n_columns={n_columns}_norm={normalize}_m={m:.2f}_c={c:.2f}"

    print(model_name)

    digital_twin = torch.load(args.model_filename, weights_only=False, map_location=device)
    digital_twin.to(device)        

    spectra = []

    print("Calculating spectra")
    for i, (image, label, description) in tqdm(enumerate(train_loader), total=25):  
        if i >= 25:                     # workaround for limited ram (pca calculation)
            break
        prediction = predict(digital_twin, device, image)
        spectra.append(prediction)

    spectra = np.concatenate(spectra, axis=0)

    pca_layer = PCAv1(n_rows, n_columns, normalize=normalize, m=m, c=c)
    pca_layer.fit(spectra, device)

    model = MultiLayerModel()

    model.add_layer(digital_twin, freeze=True)

    model.add_layer(pca_layer, freeze=True)

    mask = Maskv2()
    model.add_layer(mask)

    model.add_layer(digital_twin, freeze=True)

    regression_layer = BNLinearLayer((64,128), (10,))
    model.add_layer(regression_layer)

    model.to(device)        

    params = []
    params.append({'params': mask.parameters(), 'lr': lr_mask})
    params.append({'params': regression_layer.linear_layer.weight, 'lr': lr, "weight_decay": lambda_w})
    params.append({'params': regression_layer.linear_layer.bias, 'lr': lr})
    params.append({"params": regression_layer.bn.parameters(), "lr": lr})


    optimizer = optim.Adam(params)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    log_dir = os.path.join(LOGS_PATH, model_name)
    writer = SummaryWriter(log_dir=log_dir)

    loss_fkt = MaskLoss(nn.CrossEntropyLoss(), mask, lambda_a)

    test_acc = train_network(model, device, train_loader, test_loader, optimizer, loss_fkt, args.n_epochs, writer=writer, scheduler=scheduler, calc_acc=True, trial=trial, prune_after=args.prune_after)[3]

    if test_acc == None:
        model_name += "_pruned"

    torch.save(model, os.path.join(MODELS_PATH, f"{model_name}.pt"))
    writer.close()

    if test_acc == None:
        raise optuna.exceptions.TrialPruned()

    return test_acc


if __name__ == "__main__":
    device = get_device()
    print(f"Running on {device}")

    train_dataset = Dataset(True, args.dataset_config_path, mode="classifier")
    test_dataset = Dataset(False, args.dataset_config_path, mode="classifier")

    train_dataset = ShuffleDataset(train_dataset, args.buffer_size)
    test_dataset = ShuffleDataset(test_dataset, args.buffer_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    if "mnist" in args.dataset_config_path:
        study_name = "mnist"
    elif "cifar" in args.dataset_config_path:
        study_name = "cifar"
    elif "fashion" in args.dataset_config_path:
        study_name = "fashion"

    study_name = f"{study_name}_{args.mode}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    study_name = f"{study_name}_{np.random.randint(10000)}"

    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)

    study = optuna.create_study(storage=f"sqlite:///{LOGS_PATH}/optuna.sqlite3",
                                study_name=study_name,
                                direction="maximize",
                                load_if_exists=True)
    study.optimize(partial(objective, args=args), n_trials=100)
