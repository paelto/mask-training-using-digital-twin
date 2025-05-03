from tqdm import tqdm
import torch
import numpy as np
import cv2
from itertools import islice
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_acc(output, target):
    output_classes = torch.max(output, 1)[1]
    return (output_classes == target).float().mean()*100


def train(model, device, train_loader, optimizer, loss_fkt, calc_acc=False):
    model.train()

    epoch_loss = 0
    epoch_acc = 0

    with tqdm(total=len(train_loader)) as t:
        for data, target, description in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fkt(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if calc_acc:
                acc = get_acc(output, target)
                epoch_acc += acc
                t.set_description(f'train loss: {loss.item():.4f} acc: {acc:.1f}%')
            else:
                t.set_description(f'train loss: {loss.item():.8f}')
            t.update()

        epoch_loss /= len(train_loader)
        if calc_acc:
            epoch_acc /= len(train_loader)
            t.set_description(f'train loss: {epoch_loss:.4f} acc: {epoch_acc:.1f}%')
        else:
            t.set_description(f'train loss: {epoch_loss:.8f}')
    return epoch_loss, epoch_acc

def test(model, device, test_loader, loss_fkt, calc_acc=False):
    model.eval()

    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for data, target, description in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fkt(output, target)
                test_loss += loss.item()
                if calc_acc:
                    acc = get_acc(output, target)
                    test_acc += acc
                    t.set_description(f'test  loss: {loss.item():.4f} acc: {acc:.1f}%')
                else:
                    t.set_description(f'test  loss: {loss.item():.8f}')
                t.update()

            test_loss /= len(test_loader)
            if calc_acc:
                test_acc /= len(test_loader)
                t.set_description(f'test  loss: {test_loss:.4f} acc: {test_acc:.1f}%')
            else:
                t.set_description(f'test  loss: {test_loss:.8f}')
    return test_loss, test_acc

def predict(model, device, data, softmax=False):
    model.eval()

    with torch.no_grad():
        data = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
        data = data.to(device)
        prediction = model(data)
        if softmax:
            prediction = torch.softmax(prediction, dim=1)
        return prediction.cpu().numpy()

def plot_tensorboard(model, device, plot_data, writer, epoch):
    out_width = plot_data[0][1].shape[2] * plot_data[0][0].shape[1] // plot_data[0][0].shape[2]
    plot_image = np.zeros((0, plot_data[0][0].shape[2] + 2*out_width, 4))

    for image_in, image_out, description in plot_data:
        prediction = predict(model, device, image_in[:1])

        image_in = image_in[0].numpy()
        image_out = image_out[0].numpy()
        prediction = prediction[0]

        combined_image = np.concatenate((image_out, prediction), axis=1)
        combined_image = cv2.resize(combined_image, (2*out_width, image_in.shape[0]))
        combined_image = (combined_image - np.min(combined_image)) / (np.max(combined_image) - np.min(combined_image))
        combined_image = np.concatenate((image_in, combined_image), axis=1)
        combined_image = plt.cm.viridis(combined_image)

        plot_image = np.concatenate((plot_image, combined_image))

    writer.add_image(f"Images", plot_image, epoch+1, dataformats='HWC')
    writer.flush()

def train_network(model, device, train_loader, test_loader, optimizer, loss_fkt, n_epochs, scheduler=None, writer=None, start_epoch=0, do_plotting=False, trial=None, calc_acc=False, prune_after=0):
    if do_plotting:
        plot_data = list(islice(iter(test_loader), 5))

    for epoch in range(start_epoch, n_epochs+start_epoch):
        print(f"Epoch {epoch+1}/{n_epochs+start_epoch}")

        train_loss, train_acc = train(model, device, train_loader, optimizer, loss_fkt, calc_acc=calc_acc)
        test_loss, test_acc = test(model, device, test_loader, loss_fkt, calc_acc=calc_acc)

        if writer != None:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch+1)
            writer.add_scalar("train_loss", train_loss, epoch+1)
            writer.add_scalar("test_loss", test_loss, epoch+1)
            if calc_acc:
                writer.add_scalar("train_acc", train_acc, epoch+1)
                writer.add_scalar("test_acc", test_acc, epoch+1)
            writer.flush()
            if do_plotting:
                plot_tensorboard(model, device, plot_data, writer, epoch)

        if scheduler != None:
            scheduler.step()

        if trial != None:
            trial.report(test_acc if calc_acc else test_loss, epoch)
            if trial.should_prune() and epoch >= prune_after:
                return None, None, None, None
    
    return train_loss, train_acc, test_loss, test_acc