# Implementation of GMVAE (Mixture of Gaussian Variational AutoEncoder) trained on MNIST dataset.
# In GMVAE, the prior distribution is modeled as a mixture of Gaussians.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import struct
import matplotlib.pyplot as plt
from dataclasses import dataclass
from array import array

@dataclass
class GMVAEConfig:
    z_dim: int
    img_size: int
    num_gaussians: int
    intermediate_dim: int


class Encoder(nn.Module):
    """Encoder class implementing q(z/x) for GMVAE."""

    def __init__(self, config: GMVAEConfig):
        super().__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(config.img_size, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, 2*config.z_dim)
        )

    def forward(self, x):
        t1 = self.model(x)
        m, t_v = torch.split(t1, self.config.z_dim, dim=-1)
        v = F.softplus(t_v) + 1e-8
        return m, v
    

class Decoder(nn.Module):
    """Decoder class implementing p(x/z) for GMVAE."""

    def __init__(self, config: GMVAEConfig):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.z_dim, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, config.img_size)
        )

    def forward(self, z):
        return self.model(z)
    

class GMVAE(nn.Module):
    """Implements GMVAE model. Invokes encoder & decoder and returns the nelbo loss."""

    def __init__(self, config: GMVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.rec_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.z_prior = nn.Parameter(torch.rand(2*config.num_gaussians, config.z_dim), requires_grad=True)

    @staticmethod
    def log_normal(x, mu, var):
        """Implements log(Normal(x; mu, var))."""
        t1 = torch.sum(torch.log(var), dim=-1)
        t2 = torch.square((x - mu)) / var
        t3 = torch.sum(t2, dim=-1)
        t4 = -0.5 * (t1 + t3)
        return t4
    
    @staticmethod
    def log_sum_exp(x):
        c = torch.max(x, dim=-1, keepdim=True)
        t1 = torch.sum(torch.exp(x - c.values), dim=-1)
        t2 = torch.squeeze(c.values) + torch.log(t1)
        return t2
    
    @staticmethod
    def log_mean_gaussian(num_gaussians, x, m, v):
        """Implements log(mean(gaussian))."""
        x = torch.unsqueeze(x, dim=1)
        x_repeated = x.expand(-1, num_gaussians, -1)

        t1 = -0.5 * torch.sum(torch.log(v), dim=-1)
        t2 = -torch.sum(torch.square((x_repeated - m)) / (2*v), dim=-1)
        t3 = t1 + t2
        t4 = -torch.log(torch.tensor([num_gaussians], dtype=torch.float))
        log_mean_prob = t4 + GMVAE.log_sum_exp(t3)
        return log_mean_prob

    def forward(self, x):
        m_z, v_z = self.encoder(x)
        e = torch.normal(torch.zeros_like(m_z), torch.ones_like(v_z))
        z = m_z + e * v_z
        x_bernoulli_logits = self.decoder(z)

        rec = self.rec_loss(input=x_bernoulli_logits, target=x).sum(-1)

        # approximate KL-Div with single sample of z (Monte-carlo approximation)
        kl_t1 = self.log_normal(z, m_z, v_z)

        z_prior_m, z_prior_t1 = torch.split(self.z_prior, self.config.num_gaussians, dim=0)
        z_prior_v = F.softplus(z_prior_t1)
        kl_t2 = self.log_mean_gaussian(self.config.num_gaussians, z, z_prior_m, z_prior_v)

        kl = kl_t1 - kl_t2
        nelbo = rec + kl

        nelbo = torch.mean(nelbo, dim=0)
        kl = torch.mean(kl, dim=0)
        rec = torch.mean(rec, dim=0)
        return nelbo, rec, kl

    def forward_inference(self, batch_size):
        """Generates batch_size num of images by sampling z and running the decoder.
        First a gaussian distribution is selected at with equal prob and z is sampled from this gaussian. 
        """
        m, t1 = torch.split(self.z_prior, self.config.num_gaussians, dim=0)
        v = F.softplus(t1) + 1e-8
        k_prob = torch.tensor([1.0 / self.config.num_gaussians] * self.config.num_gaussians)
        k_dist = torch.distributions.categorical.Categorical(probs=k_prob)
        k = k_dist.sample(sample_shape=[batch_size])

        e_m = torch.zeros(batch_size, self.config.z_dim)
        v_m = torch.ones(batch_size, self.config.z_dim)
        e = torch.normal(e_m, v_m)
        z = m[k, :] + e*v[k, :]
        x_logits = self.decoder(z)
        x_bernoulli_probs = F.sigmoid(x_logits)
        x = torch.bernoulli(x_bernoulli_probs)
        return x


class MNISTDataset(Dataset):
    def __init__(self, img_path):
        super().__init__()
        images = []

        with open(img_path, 'rb') as f2:
            _, num_images, rows, cols = struct.unpack(">IIII", f2.read(16))

            all_images = array("B", f2.read())

            for i in range(num_images):
                images.append([0]*rows*cols)

            for i in range(num_images):
                image = all_images[i*rows*cols : (i+1)*rows*cols]
                images[i][:] = image

        self.processed_image_tensor = self.process_images(images)

    def __len__(self):
        return len(self.processed_image_tensor)
    
    def __getitem__(self, index):
        return self.processed_image_tensor[index]

    def process_images(self, images):
        """
        Processes images to create float tensor whose values lie between 0 & 1.
        An Image is converted to a float tensor, scaled and the values are sampled
        from a bernoulli dist, so that all values in image tensor are 0 or 1.
        """
        bernoulli_param = torch.tensor(images, dtype=torch.float) / 255.0
        x = torch.bernoulli(bernoulli_param)
        return x


def sample_and_plot(model: GMVAE, epoch, batch):
    x = model.forward_inference(batch_size=200)
    images = x.reshape(200, 28, 28)
    figs, axs = plt.subplots(nrows=10, ncols=20, figsize=(28,28))

    idx = 0
    for ax in axs.reshape(-1):
        ax.imshow(images[idx].detach().numpy())
        idx += 1

    plt.tight_layout()
    plt.savefig(f'e{epoch}_b{batch}_gmvae.png')
    

def train(model: GMVAE, config: GMVAEConfig, batch_size, num_epoch, img_path):
    dataset = MNISTDataset(img_path)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(num_epoch):
        for batch, x in enumerate(dataloader):
            optimizer.zero_grad()
            nelbo, rec, kl = model(x)
            
            if batch % 100 == 0:
                print(f'epoch: {epoch}, batch: {batch} || nelbo: {nelbo}, rec: {rec}, kl: {kl}')

            nelbo.backward()
            optimizer.step()


def main():
    config = GMVAEConfig(
        z_dim=10,
        img_size=784,
        num_gaussians=500,
        intermediate_dim=300
    )
    model = GMVAE(config)
    img_path = '/kaggle/input/mnist-dataset/train-images.idx3-ubyte'

    print('starting training !')
    train(model, config, batch_size=5, num_epoch=2, img_path=img_path)
    print('training complete !!')
    sample_and_plot(model, epoch=777, batch=777)


if __name__ == "__main__":
    main()

