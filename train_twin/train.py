from model_architectures.UNetModelv4 import UNetModelv4
from tools.dataloader import Dataset, ShuffleDataset
from tools.pytorch_essentials import train_network, get_device
from tools.helper import to_full_path

import argparse
import datetime
import optuna
from functools import partial
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

LOGS_PATH = "logs"
MODELS_PATH = "models"

def objective(trial, args):
    dropout = trial.suggest_categorical('dropout', [True, False])
    if dropout:
        dropout_p = trial.suggest_float('dropout_p', 0.2, 0.8, step=0.1)
    else:
        dropout_p = 0.0

    norm = trial.suggest_categorical('norm', ["batch_norm", "layer_norm"])
#    if norm  == "layer_norm":
#        layer_norm_par = trial.suggest_categorical('layer_norm_par', [True, False])
    layer_norm_par = False

    n = trial.suggest_int('n', 5, 7)
    if n == 7:
        f_beg = trial.suggest_int('f_beg', 10, 20)
        f_end = trial.suggest_int('f_end', 10, 20) 
    else:    
        f_beg = trial.suggest_int('f_beg', 30, 60)
        f_end = trial.suggest_int('f_end', 30, 60) 
    f_mid = trial.suggest_int('f_mid', 500, 800)
    f_fin = f_end // 2

    gamma = trial.suggest_float('gamma', 0.7, 0.99)

    model = UNetModelv4(n, f_beg, f_mid, f_end, f_fin, dropout, dropout_p, norm, layer_norm_par).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([np.prod(p.size()) for p in model_parameters])
    model_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name += f" lr={args.learning_rate} gamma={gamma:.2e} batch_size={args.batch_size} unetv4 n={n} f_beg={f_beg} f_mid={f_mid} f_end={f_end} f_fin={f_fin} dropout_p={dropout_p:.1} norm={norm} ln_p={layer_norm_par} n_param={n_parameters//1000}k"

    print(model_name)

    log_dir = os.path.join(LOGS_PATH, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    writer = SummaryWriter(log_dir=log_dir)

    test_loss = train_network(model, device, train_loader, test_loader, optimizer, nn.MSELoss(), args.n_epochs, writer=writer, scheduler=scheduler, do_plotting=True, trial=trial)[2]

    if test_loss == None:
        model_name += "_pruned"

    model_filename = os.path.join(MODELS_PATH, f"{model_name}.pt")
    torch.save(model, to_full_path(model_filename))
    writer.close()

    if test_loss == None:
        raise optuna.exceptions.TrialPruned()
        
    return test_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training digital twin")
    parser.add_argument('dataset_config_path', type=str, help="Path to the dataset configuration file")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument('--buffer_size', type=int, default=5000, help="Buffer size (default: 5000)")
    parser.add_argument('--n_epochs', type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (default: 0.001)")
#    parser.add_argument('--gamma', type=float, default=0.8, help="Gamma for learning rate decay (default: 0.8)")

    args = parser.parse_args()

    device = get_device()
    print(f"Running on {device}")

    train_dataset = Dataset(True, args.dataset_config_path)
    test_dataset = Dataset(False, args.dataset_config_path)

    train_dataset = ShuffleDataset(train_dataset, args.buffer_size)
    test_dataset = ShuffleDataset(test_dataset, args.buffer_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    study = optuna.create_study(storage=f"sqlite:///{LOGS_PATH}/optuna.sqlite3",
                                study_name=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                direction="minimize")
    study.optimize(partial(objective, args=args), n_trials=200)