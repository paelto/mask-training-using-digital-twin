import torch
import torch.nn as nn

import sys
sys.path.append('.')
from mask_training.models import Maskv2

class MultiMaskModel(nn.Module):
    def __init__(self, digital_twin, bn=True, num_masks=10):
        super().__init__()
        self.bn = bn
        self.num_masks = num_masks
        self.masks = nn.ModuleList([Maskv2() for _ in range(num_masks)])
        self.digital_twin = digital_twin
        self.regressors = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(8192) if bn else nn.Identity(),
                nn.Linear(8192, 1)
            )
            for _ in range(num_masks)
        ])

        for param in self.digital_twin.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = []
        for i in range(self.num_masks):
            masked_x = self.masks[i](x)
            twin_out = self.digital_twin(masked_x)
            twin_out = twin_out.view(twin_out.size(0), -1)
            output = self.regressors[i](twin_out)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)

        return outputs
    
    def train(self, mode=True):
        super().train(mode=mode)
        self.digital_twin.eval()
        return self


class MultiMaskLoss(nn.Module):
    def __init__(self, loss_fkt, model, lambda_a):
        super().__init__()
        self.loss_fkt = loss_fkt
        self.model = model
        self.lambda_a = lambda_a

    def forward(self, output, target):
        loss = self.loss_fkt(output, target)
        
        reg = 0.0
        for mask in self.model.masks:

            reg += torch.mean((1 - torch.sigmoid(mask.a)) ** 2) * 900
        
        return loss + self.lambda_a * reg




