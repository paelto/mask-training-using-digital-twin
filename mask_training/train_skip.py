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
    for mode in ["mask_skipv1"]:
        subparser = subparsers.add_parser(mode)
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
import sys
sys.path.append('.')

from tools.dataloader import Dataset, ShuffleDataset
from tools.pytorch_essentials import train_network, get_device
from mask_training.models import Maskv2, MultiLayerModel, BNLinearLayer, Combinev1
from mask_training.losses import MaskLoss

LOGS_PATH = "logs"
MODELS_PATH = "mask_training/models"

def objective(trial, args):
    lr = trial.suggest_float('lr', 3e-4, 5e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.7, 0.99)
    lambda_w = trial.suggest_float('lambda_w', 1e-7, 1e-4, log=True)

    lr_mask = trial.suggest_float('lr_mask', 5e-4, 1e-1, log=True)
    lambda_a = trial.suggest_float('lambda_a', 1e-7, 1e-2, log=True)

#    models = {
#        "20250131-062221": "mask_training/models_old3/mnist_mask_pdt_20250131-062221_mask_lr_mask=4.88e-02_gamma=8.65e-01_lambda_a=8.44e-06_lambda_pull=1.06e-23.pt",
#        "20250130-221201": "mask_training/models_old3/mnist_mask_pdt_20250130-221201_mask_lr_mask=2.88e-02_gamma=8.06e-01_lambda_a=4.95e-04_lambda_pull=3.73e-21.pt",
#        "20250201-215833": "mask_training/models_no_norm/mnist_mask_pdt_no_norm_20250201-215833_mask_lr_mask=1.15e-02_gamma=8.07e-01_lambda_a=5.30e-04_lambda_pull=4.77e+00.pt",
#        "20250201-224758": "mask_training/models_no_norm/mnist_mask_pdt_no_norm_20250201-224758_mask_lr_mask=2.17e-03_gamma=9.45e-01_lambda_a=3.19e-05_lambda_pull=1.59e+01.pt",
#        "20250110-052820": "mask_training/best_base/mnist_maskv2_20250110-052820_lr=2.72e-03_gamma=9.48e-01_lambda_w=1.02e-05_mask_lr_mask=5.94e-03_lambda_a=8.28e-06.pt"
#    }
#    base_name = trial.suggest_categorical('base_name', list(models.keys()))
#    base_filename = models[base_name]
    base_name = "no_mask"

    crop = trial.suggest_categorical("crop", [0, 1])
    normalize = trial.suggest_categorical("normalize", [0, 1])
    side_by_side = trial.suggest_categorical("side_by_side", [0, 1])
    
    if normalize:
        m1 = trial.suggest_float("m1", 0.1, 0.4)
        c1 = trial.suggest_float("c1", 0.1, 0.7)
    else:
        m1 = trial.suggest_float("m1", 2.5, 10)
        c1 = trial.suggest_float("c1", -0.3, 0.3)

    m2 = trial.suggest_float("m2", 0.5, 2)
    c2 = trial.suggest_float("c2", -0.3, 0.3)

    model_name = "mnist" if "mnist" in args.dataset_config_path else "cifar" if "cifar" in args.dataset_config_path else "fashion" if "fashion" in args.dataset_config_path else "unknown"

    model_name = f"{model_name}_{args.mode}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model_name = f"{model_name}_lr={lr:.2e}_gamma={gamma:.2e}_lambda_w={lambda_w:.2e}"
    model_name = f"{model_name}_lr_mask={lr_mask:.2e}_lambda_a={lambda_a:.2e}"
    model_name = f"{model_name}_base={base_name}_crop={crop}_norm={normalize}_sbs={side_by_side}"
    model_name = f"{model_name}_m1={m1:.2e}_c1={c1:.2e}_m2={m2:.2e}_c2={c2:.2e}"

    print(model_name)

#    base_model = torch.load(base_filename, weights_only=False, map_location=device)

    model = MultiLayerModel()

#    for layer in base_model.layers[:2]:
#        model.add_layer(layer, freeze=True)
    digital_twin = torch.load("models/model1.pt", weights_only=False, map_location=device)
    model.add_layer(digital_twin)

    combine_layer=Combinev1(crop, side_by_side, normalize, m1, c1, m2, c2)
    model.add_layer(combine_layer)

    mask = Maskv2()
    model.add_layer(mask)

#    digital_twin = base_model.layers[1]
    model.add_layer(digital_twin)

    regression_layer = BNLinearLayer((64,128), (10,))
    model.add_layer(regression_layer)

    model.to(device)        

    torch.save(model, os.path.join(MODELS_PATH, f"{model_name}.pt"))


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
