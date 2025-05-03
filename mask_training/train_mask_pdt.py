import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for mask training")
    parser.add_argument('dataset_config_path', type=str, help="Path to the dataset configuration file")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument('--buffer_size', type=int, default=5000, help="Buffer size (default: 5000)")
    parser.add_argument('--n_epochs', type=int, default=10, help="Number of epochs (default: 10)")

    subparsers = parser.add_subparsers(dest="mode", required=True)
    parsers = []
    for mode in ["mask_pdt"]:
        subparser = subparsers.add_parser(mode)
        parsers.append(subparser)
        subparser.add_argument('model_filename', type=str, help="Filename of base model")

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
from tools.pytorch_essentials import train_network, get_device, test, train
from mask_training.models import Maskv2, NormalizeLayer, MultiLayerModel
from mask_training.losses import MaskLoss, PdtLoss

LOGS_PATH = "logs_no_norm"
MODELS_PATH = "mask_training/models_no_norm"

def objective(trial, args):
    lr_mask = trial.suggest_float('lr_mask', 1e-4, 5e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.95)
    lambda_a = trial.suggest_float('lambda_a', 1e-6, 1e-2, log=True)
    lambda_pull = trial.suggest_float('lambda_pull', 1E-2, 1E6, log=True)


    if "mnist" in args.dataset_config_path:
        model_name = "mnist"
    elif "cifar" in args.dataset_config_path:
        model_name = "cifar"

#    model_name = f"{model_name}_{args.mode}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model_name = f"{model_name}_{args.mode}_no_norm_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model_name = f"{model_name}_mask_lr_mask={lr_mask:.2e}_gamma={gamma:.2e}_lambda_a={lambda_a:.2e}_lambda_pull={lambda_pull:.2e}"

    print(model_name)

    model = MultiLayerModel()

    mask = Maskv2()        
    model.add_layer(mask)

    digital_twin = torch.load(args.model_filename, weights_only=False, map_location=device)    
    model.add_layer(digital_twin, freeze=True)

#    norm_layer = NormalizeLayer()
#    model.add_layer(norm_layer)

    params = [{'params': mask.parameters(), 'lr': lr_mask}]

    model.to(device)        

    optimizer = optim.Adam(params)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    log_dir = os.path.join(LOGS_PATH, model_name)
    writer = SummaryWriter(log_dir=log_dir)

    loss_fkt = PdtLoss(lambda_pull, 5E-6, 1E-3)
    loss_fkt = MaskLoss(loss_fkt, mask, lambda_a)

    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch+1}/{args.n_epochs}")

        train_loss = train(model, device, train_loader, optimizer, loss_fkt)[0]
        print("pull_loss")
        loss_pull = test(model, device, test_loader, PdtLoss(1E2, 0, 1E-3))[0]
        print("push_loss")
        loss_push = test(model, device, test_loader, PdtLoss(0, 5E-6, 1E-3))[0]

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch+1)
        writer.add_scalar("train_loss", train_loss, epoch+1)
        writer.add_scalar("loss_pull", loss_pull, epoch+1)
        writer.add_scalar("loss_push", loss_push, epoch+1)
        writer.flush()

        scheduler.step()
    
    torch.save(model, os.path.join(MODELS_PATH, f"{model_name}.pt"))

    return loss_pull + loss_push


if __name__ == "__main__":
    device = get_device()
    print(f"Running on {device}")

    train_dataset = Dataset(True, args.dataset_config_path, mode="classifier_spectrum" if args.mode == "spectrum" else "classifier")
    test_dataset = Dataset(False, args.dataset_config_path, mode="classifier_spectrum" if args.mode == "spectrum" else "classifier")

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
                                direction="minimize")
    study.optimize(partial(objective, args=args), n_trials=100)
