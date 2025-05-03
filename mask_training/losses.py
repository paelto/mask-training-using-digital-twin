import torch
import torch.nn as nn

class MaskLoss(nn.Module):
    def __init__(self, loss_fkt, model, lambda_a):
        super().__init__()
        self.loss_fkt = loss_fkt
        self.model = model
        self.lambda_a = lambda_a

    def forward(self, output, target):
        loss = self.loss_fkt(output, target)
        reg = self.lambda_a * torch.mean((1 - torch.sigmoid(self.model.a)) ** 2) * 900
        return loss + reg

class PdtLoss(nn.Module):
    def __init__(self, lambda_pull, lambda_push, epsilon):
        super().__init__()
        self.lambda_pull = lambda_pull
        self.lambda_push = lambda_push
        self.epsilon = epsilon

    def forward(self, spectra, targets):
        average_spectra = []

        for value in range(10):
            spectra_for_value = spectra[targets == value]
            
            if len(spectra_for_value) > 0:
                average_spectra.append(spectra_for_value.mean(dim=0).detach())
            else:
                average_spectra.append(spectra.mean(dim=0).detach())
            
        loss_pull = 0
        loss_push = 0

        for spectrum, target in zip(spectra, targets):
            for class_label in range(10):
                diff = torch.abs(spectrum - average_spectra[class_label])
                if class_label == target:
                    loss_pull += self.lambda_pull * torch.mean(diff ** 2)
                else:
                    loss_push += self.lambda_push / (torch.sqrt(torch.mean(diff ** 2)) + self.epsilon)

        return loss_pull + loss_push

