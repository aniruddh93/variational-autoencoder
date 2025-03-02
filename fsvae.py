# Implementation of Fully-Supervised Variational Autoencoder (FSVAE) trained on SVHN dataset.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from scipy import io
import matplotlib.pyplot as plt

from dataclasses import dataclass

@dataclass
class FSVAEConfig:
    y_dim: int
    z_dim: int
    intermediate_dim: int
    img_size: int
    kl_penalty: int


class Encoder(nn.Module):
    """Implements FSVAE encoder modeling q(z/x, y)"""

    def __init__(self, config: FSVAEConfig):
        super().__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(config.img_size + config.y_dim, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, 2*config.z_dim),
        )

    def forward(self, x, y):
        xy = torch.concat((x, y), dim=-1)
        out = self.model(xy)
        mu, t1 = torch.split(out, self.config.z_dim, dim=-1)
        v = F.softplus(t1) + 1e-8
        return mu, v
    

class Decoder(nn.Module):
    """Implements FSVAE decoder modeling p(x/z, y)"""

    def __init__(self, config: FSVAEConfig):
        super().__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(config.z_dim + config.y_dim, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, config.img_size),
        )

    def forward(self, z, y):
        zy = torch.cat((z, y), dim=-1)
        mu = self.model(zy)
        v = torch.ones(z.shape[0], self.config.img_size, dtype=torch.float, requires_grad=False) * 0.1
        return mu, v
    

class FSVAE(nn.Module):
    """Implementation of FSVAE."""

    def __init__(self, config: FSVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.rec_loss = nn.GaussianNLLLoss(reduction="none")

    def forward(self, x, y):
        """Runs both encoder & decoder and returns rec loss, kl div and nelbo loss"""
        m_z, v_z = self.encoder(x, y)
        e = torch.normal(mean=torch.zeros_like(m_z), std=torch.ones_like(v_z))
        z = m_z + e * torch.sqrt(v_z)
        m_x, m_v = self.decoder(z, y)

        rec = self.rec_loss(input=m_x, target=x, var=m_v).sum(dim=-1)

        # simplified KL term, calculated analytically
        kl_t = 0.5 * (-torch.log(v_z) - 1 + v_z + (m_z * m_z))
        kl = torch.sum(kl_t, dim=-1)

        nelbo = rec + (self.config.kl_penalty * kl)
        nelbo = nelbo.mean(dim=0)
        kl = kl.mean(dim=0)
        rec = rec.mean(dim=0)
        return nelbo, kl, rec
    
    def forward_inference(self, y):
        """Samples z, invokes only the decoder model and returns mean and variance to generate new sample x."""
        batch_size = y.shape[0]
        z = torch.normal(mean=torch.zeros(batch_size, self.config.z_dim), std=torch.ones(batch_size, self.config.z_dim))
        m_x, v_x = self.decoder(z, y)
        return m_x, v_x


class SVHNDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        mat = io.loadmat(dataset_path)
        self.y = np.squeeze(mat['y'])
        self.x = np.transpose(mat['X'], (3, 0, 1, 2))

    def __getitem__(self, index):
        X = torch.tensor(self.x[index], dtype=torch.float) / 255.0
        y_val = self.y[index] if self.y[index] != 10 else 0
        Y = torch.tensor(y_val, dtype=torch.int)
        return X, Y
    
    def __len__(self):
        return len(self.y)


def plot(sampled_x, batch, epoch):
    images = sampled_x.reshape(10, 32, 32, 3)
    figs, axs = plt.subplots(nrows=10, ncols=1, figsize=(28,28))
    
    idx = 0
    for ax in axs.reshape(-1):
        ax.imshow(images[idx].detach().numpy())
        idx += 1

    plt.tight_layout()
    plt.savefig(f'trained_samples_e{epoch}_b{batch}.png')
    plt.close()
    

def train(model: FSVAE, config: FSVAEConfig, dataset_path, batch_size, num_epoch):
    dataset = SVHNDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(num_epoch):
        for batch, (img, target) in enumerate(dataloader):
            t1 = torch.eye(config.y_dim)
            target_emb = t1[target, :]
            img = img.reshape(batch_size, config.img_size)
            nelbo, kl, rec = model(img, target_emb)

            optimizer.zero_grad()
            nelbo.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f'epoch {epoch}, batch {batch} || nelbo: {nelbo}, kl: {kl}, rec: {rec}')

            # sample every 1000 iterations.
            if batch % 500 == 0:
                m_x, v_x = model.forward_inference(t1)
                clipped_m_x = torch.clip(m_x, min=0.0, max=1.0)
                
                # modeling continuos image data is quite challenging.
                # common heuristic is to sample clip(u(z, y)) rather than p(x/y,z)
                plot(clipped_m_x, batch, epoch)
            
            
    

def main():
    config = FSVAEConfig(
        y_dim=10,
        z_dim=10,
        intermediate_dim=500,
        img_size=3072,
        kl_penalty=1
    )

    model = FSVAE(config)
    dataset_path = '/kaggle/input/svhn-custom/extra_32x32.mat'

    print('starting training !')
    train(model, config, dataset_path, batch_size=10, num_epoch=1)
    print('training complete !!')


if __name__ == "__main__":
    main()

