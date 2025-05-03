import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import inspect

class Identity(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.input_shape = self.output_shape = shape
        
    def forward(self, x):
        return x

class ScaleLayer(nn.Module):
    def __init__(self, mult, offset):
        super().__init__()
        self.input_shape = (64, 128)
        self.output_shape = (100, 100)
        self.mult = mult
        self.offset = offset

    def forward(self, x):
        x = x[:, 10:60, 20:116]
        assert x.shape[1] <= 96 and x.shape[2] <= 96
        x = x.unsqueeze(1)
        x = F.interpolate(x, size=(96, 96), mode='nearest')
        x = x * self.mult + self.offset
        x = torch.clamp(x, min=0, max=1)
        x = F.pad(x, (2, 2, 2, 2), value=1)
        x = x.squeeze(1)
        return x


class Maskv1(nn.Module):
    input_shape = (100,100)
    output_shape = (100,100)

    def __init__(self):
        super().__init__()
        self.a = Parameter(torch.rand((30, 30)))        # fix setting random twice
        self.b = Parameter(torch.rand((30, 30)))
        torch.nn.init.uniform_(self.a, a=0, b=2)
        torch.nn.init.uniform_(self.b, a=-3, b=3)

        self.scaling_matrix = nn.Parameter(torch.ones(3, 3), requires_grad=False)

    def get_scaled_mask(self):
        a_scaled = torch.sigmoid(self.a)
        b_scaled = torch.sigmoid(self.b)

        a_scaled = torch.kron(a_scaled, self.scaling_matrix)
        b_scaled = torch.kron(b_scaled, self.scaling_matrix)

        a_scaled = F.pad(a_scaled, (5, 5, 5, 5), value=1)
        b_scaled = F.pad(b_scaled, (5, 5, 5, 5), value=1)

        return a_scaled, b_scaled

    def forward(self, x):
        a_scaled, b_scaled = self.get_scaled_mask()

        return x * a_scaled + b_scaled * (1 - a_scaled)    

class Maskv2(nn.Module):
    input_shape = (100,100)
    output_shape = (100,100)

    def __init__(self):
        super().__init__()
        self.a = Parameter(torch.rand((90, 90)))
        self.b = Parameter(torch.rand((90, 90)))
        torch.nn.init.uniform_(self.a, a=0, b=2)
        torch.nn.init.uniform_(self.b, a=-3, b=3)

    def get_scaled_mask(self):
        a_scaled = torch.sigmoid(self.a)
        b_scaled = torch.sigmoid(self.b)

        a_scaled = F.pad(a_scaled, (5, 5, 5, 5), value=1)
        b_scaled = F.pad(b_scaled, (5, 5, 5, 5), value=1)

        return a_scaled, b_scaled

    def forward(self, x):
        a_scaled, b_scaled = self.get_scaled_mask()

        return x * a_scaled + b_scaled * (1 - a_scaled)  

class BNLinearLayer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.bn = nn.BatchNorm1d(np.prod(input_shape))
        self.linear_layer = nn.Linear(np.prod(input_shape), np.prod(output_shape))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.bn(x)
        x = self.linear_layer(x)
        x = x.view(-1, *self.output_shape)
        return x

class RegressionLayer(BNLinearLayer):
    def __init__(self):
        super().__init__((64,128), (10,))

    def forward(self, x):
        self.input_shape = (64, 128)
        self.output_shape = (10,)
        return super().forward(x)


class NormalizeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = self.output_shape = (64, 128)
        self.bn = nn.BatchNorm1d(8192, affine=False)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.bn(x)
        x = x.view(-1, 64, 128)        
        return x

class MultiLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.freeze_indices = []
        self.input_shape = None
        self.output_shape = None

    def add_layer(self, layer, freeze=False):
        if len(self.layers) == 0:
            self.input_shape = layer.input_shape
        else:
            if self.layers[-1].output_shape != layer.input_shape:
                raise ValueError(f"Output shape {self.layers[-1].output_shape} does not match input shape {layer.input_shape}")

        if freeze:
            self.freeze_indices.append(len(self.layers))
            for param in layer.parameters():
                param.requires_grad = False

        self.layers.append(layer)
        self.output_shape = layer.output_shape

    def forward(self, input_image):
        x = input_image
        for layer in self.layers:
            if 'input_image' in inspect.signature(layer.forward).parameters:
                x = layer(x, input_image=input_image)
            else:
                x = layer(x)
        return x

    def train(self, mode=True):
        super().train(mode=mode)
        for i in self.freeze_indices:
            self.layers[i].eval()
        return self
    
class CombinedModel(nn.Module):
    def __init__(self, input_model, digital_twin, output_model):
        super().__init__()
        assert input_model.output_shape == digital_twin.input_shape
        assert digital_twin.output_shape == output_model.input_shape
        self.input_shape = input_model.input_shape
        self.output_shape = output_model.output_shape
        self.input_model = input_model
        self.digital_twin = digital_twin
        self.output_model = output_model

        for param in digital_twin.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.input_model(x)
        x = self.digital_twin(x)
        x = self.output_model(x)
        return x
    
    def train(self, mode=True):
        super().train(mode=mode)
        self.digital_twin.eval()
        return self

    @property
    def layers(self):
        return [self.input_model, self.digital_twin, self.output_model]

    @property
    def freeze_indices(self):
        return [1]
    
class Combinev1(nn.Module):
    input_shape = (64,128)
    output_shape = (100,100)

    def __init__(self, crop, side_by_side, normalize, m1, c1, m2, c2):
        super().__init__()

        self.crop = crop
        self.side_by_side = side_by_side
        self.normalize = normalize
        self.m1 = m1
        self.c1 = c1
        self.m2 = m2
        self.c2 = c2

        if side_by_side:
            self.spectrum_size = (48, 96)
        else:
            self.spectrum_size = (96, 96)

        self.bn = nn.BatchNorm1d(np.prod(self.spectrum_size), affine=False)
        
    def forward(self, x, input_image):
        if self.crop:
            x = x[:, 10:60, 20:116]

        x = x.unsqueeze(1)
        x = F.interpolate(x, size=self.spectrum_size, mode='bilinear')

        y = input_image.unsqueeze(1)
        y = F.interpolate(y, size=self.spectrum_size, mode='bilinear')

        if self.normalize:
            x = torch.flatten(x, start_dim=1)
            x = self.bn(x)
            x = x.view(-1, 1, *self.spectrum_size)

        x = self.m1 * x + self.c1
        y = self.m2 * y + self.c2

        if self.side_by_side:
            x = torch.cat((x, y), dim=2)
        else:
            x = (x + y)/2

        x = F.pad(x, (2, 2, 2, 2), value=1)
        x = x.squeeze(1)
        x = torch.clamp(x, min=0, max=1)

        return x
    
    

if __name__ == "__main__":
    import torchsummary
    import sys
    sys.path.append('.')
    from model_architectures.UNetModelv3 import UNetModelv3

    input_model = Maskv1()
    digital_twin = UNetModelv3(5, 10, 20, 10, 5, False)
    output_model = RegressionLayer()

    model = CombinedModel(input_model, digital_twin, output_model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{param.shape} {name}")

    torchsummary.summary(model, model.input_shape)