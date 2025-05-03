import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

class PCAv1(nn.Module):
    input_shape = (64, 128)
    output_shape = (100, 100)

    def __init__(self, n_rows, n_columns, normalize=False, m=1, c=0.5):
        super().__init__()

        self.n_rows = n_rows
        self.n_columns = n_columns
        self.normalize = normalize
        self.m = m
        self.c = c
        self.row_indices = []
        self.col_indices = []

        available_positions = [(i, j) for i in range(n_rows) for j in range(n_columns)]
        for _ in range(n_rows * n_columns):
            best_spot = max(available_positions, key=lambda pos: min(((pos[0] - p[0])**2 + (pos[1] - p[1])**2 for p in zip(self.row_indices, self.col_indices)), default=pos[0]==n_rows//2 and pos[1]==n_columns//2))
            available_positions.remove(best_spot)
            self.row_indices.append(best_spot[0])
            self.col_indices.append(best_spot[1])

    def fit(self, spectra, device):
        spectra = torch.tensor(spectra, device=device)
        spectra = spectra.reshape(spectra.shape[0], -1)

        if self.normalize:
            self.mean = torch.mean(spectra, dim=0)
            self.std = torch.std(spectra, dim=0)
            self.std = torch.where(self.std == 0, torch.ones_like(self.std, device=device), self.std)

            spectra = (spectra - self.mean) / self.std
        
        self.pca = PCA(n_components=self.n_rows * self.n_columns)
        self.pca.fit_transform(spectra.cpu().numpy())
        self.components = torch.tensor(self.pca.components_, device=device)
        self.mean_pca = torch.tensor(self.pca.mean_, device=device)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)

        if self.normalize:
            x = (x - self.mean) / self.std

        pca_components = torch.matmul(x - self.mean_pca, self.components.T).float()

        x = torch.zeros(len(pca_components), self.n_rows, self.n_columns, device=x.device)
        x[:, self.row_indices, self.col_indices] = pca_components

        if not hasattr(self, 'm'):  # workaround
            x = x + 0.5
        else:
            x = self.m * x + self.c

        x = torch.clamp(x, min=0, max=1)
        x = torch.kron(x, torch.ones((90//self.n_rows, 90//self.n_columns), device=x.device))

        pad_1 = (self.output_shape[0] - x.shape[1])//2
        pad_2 = self.output_shape[0] - pad_1 - x.shape[1]
        pad_3 = (self.output_shape[1] - x.shape[2])//2
        pad_4 = self.output_shape[1] - pad_3 - x.shape[2]
        x = F.pad(x, pad=(pad_3, pad_4, pad_1, pad_2), value=1)

        return x