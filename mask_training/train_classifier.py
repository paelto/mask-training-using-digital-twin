import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for mask training")
    parser.add_argument('dataset_config_path', type=str, help="Path to the dataset configuration file")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument('--buffer_size', type=int, default=5000, help="Buffer size (default: 5000)")
    parser.add_argument('--n_epochs', type=int, default=10, help="Number of epochs (default: 10)")

    subparsers = parser.add_subparsers(dest="mode", required=True)
    parser1 = subparsers.add_parser("spectrum")
    parser2 = subparsers.add_parser("twin")
    parser2.add_argument('model_filename', type=str, help="Filename of digital twin")
    parser3 = subparsers.add_parser("maskv1")
    parser3.add_argument('model_filename', type=str, help="Filename of digital twin")
    parser4 = subparsers.add_parser("maskv2")
    parser4.add_argument('model_filename', type=str, help="Filename of digital twin")

    args = parser.parse_args()

import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from torch.utils.tensorboard import SummaryWriter
import optuna
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import sys
sys.path.append('.')

from tools.dataloader import Dataset, ShuffleDataset
from tools.pytorch_essentials import train_network, get_device
from mask_training.models import Maskv1, Maskv2, RegressionLayer, CombinedModel, MaskLoss, Identity

LOGS_PATH = "logs"
MODELS_PATH = "mask_training/models"

def objective(trial, args):
    lr = trial.suggest_float('lr', 3e-4, 5e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.95)
    lambda_w = trial.suggest_float('lambda_w', 1e-5, 3e-4, log=True)
    if args.mode[:4] == "mask":
        lr_mask = trial.suggest_float('lr_mask', 8e-4, 6e-3, log=True)
        lambda_a = trial.suggest_float('lambda_a', 1e-6, 1e-2, log=True)

    if "mnist" in args.dataset_config_path:
        model_name = "mnist"
    elif "cifar" in args.dataset_config_path:
        model_name = "cifar"
    model_name = f"{model_name}_{args.mode}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model_name = f"{model_name}_lr={lr:.2e}_gamma={gamma:.2e}_lambda_w={lambda_w:.2e}"
    if args.mode[:4] == "mask":
        model_name = f"{model_name}_mask_lr_mask={lr_mask:.2e}_lambda_a={lambda_a:.2e}"

    print(model_name)

    if args.mode == "spectrum":
        input_model = Identity((64, 128))
        digital_twin = Identity((64, 128))
    elif args.mode == "twin":
        input_model = Identity((100, 100))
        digital_twin = torch.load(args.model_filename, weights_only=False, map_location=device)
    elif args.mode == "maskv1":
        input_model = Maskv1()
        digital_twin = torch.load(args.model_filename, weights_only=False, map_location=device)
    elif args.mode == "maskv2":
        input_model = Maskv2()
        digital_twin = torch.load(args.model_filename, weights_only=False, map_location=device)

    output_model = RegressionLayer()
    model = CombinedModel(input_model, digital_twin, output_model)
    model.to(device)        

    params = [
        {"params": model.output_model.bn.parameters(), "weight_decay": 0, "lr": lr},
        {"params": model.output_model.linear_layer.weight, "weight_decay": lambda_w, "lr": lr},
        {"params": model.output_model.linear_layer.bias, "weight_decay": 0, "lr": lr}
    ]

    if args.mode[:4] == "mask":
        params.append({"params": model.input_model.parameters(), "lr": lr_mask})

    optimizer = optim.Adam(params)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    log_dir = os.path.join(LOGS_PATH, model_name)
    writer = SummaryWriter(log_dir=log_dir)

    if args.mode[:4] == "mask":
        loss_fkt = MaskLoss(nn.CrossEntropyLoss(), model.input_model, lambda_a)
    else:
        loss_fkt = nn.CrossEntropyLoss()

    test_acc = train_network(model, device, train_loader, test_loader, optimizer, loss_fkt, args.n_epochs, writer=writer, scheduler=scheduler, calc_acc=True)[3]

    torch.save(model, os.path.join(MODELS_PATH, f"{model_name}.pt"))

    return test_acc


if __name__ == "__main__":
    device = get_device()
    print(f"Running on {device}")

    train_dataset = Dataset(True, args.dataset_config_path, mode="classifier_spectrum" if args.mode == "spectrum" else "classifier")
    test_dataset = Dataset(False, args.dataset_config_path, mode="classifier_spectrum" if args.mode == "spectrum" else "classifier")

    if hasattr(train_dataset, 'selector'): # temporary
        test_dataset.selector = train_dataset.selector

    train_dataset = ShuffleDataset(train_dataset, args.buffer_size)
    test_dataset = ShuffleDataset(test_dataset, args.buffer_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)


    if "mnist" in args.dataset_config_path:
        study_name = "mnist"
    elif "cifar" in args.dataset_config_path:
        study_name = "cifar"
    study_name = f"{study_name}_{args.mode}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)
    study = optuna.create_study(storage=f"sqlite:///{LOGS_PATH}/optuna.sqlite3",
                                study_name=study_name,
                                direction="maximize")
    study.optimize(partial(objective, args=args), n_trials=30)
