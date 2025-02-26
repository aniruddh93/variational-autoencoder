# Implementation of vanilla VAE trained on MNIST dataset.

import struct
from array import array
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass

@dataclass
class VAEConfig:
    z_dim: int
    intermediate_dim: int
    image_size: int
    kl_penalty: int


class Encoder(nn.Module):
    """Implements VAE encoder that outputs mean & variance for the normal distribution modelling p(z/x)"""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.Linear(config.image_size, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, 2*config.z_dim)
        )

    def forward(self, x):
        out = self.model(x)
        mu, t = torch.split(out, self.config.z_dim, dim=-1)
        sigma = F.softplus(t) + 1e-8    # make variance positive without changing magnitude too much
        return mu, sigma
    

class Decoder(nn.Module):
    """Implements VAE decoder that outputs bernoulli parameter for p(x/z) distribution."""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.Linear(config.z_dim, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, config.intermediate_dim),
            nn.ELU(),
            nn.Linear(config.intermediate_dim, config.image_size)
        )

    def forward(self, z):
        return self.model(z)
    

class MNISTDataset(Dataset):
    """Loads MNIST images."""

    def __init__(self, img_path, label_path):
        super().__init__()
        self.img_path = img_path
        self.label_path = label_path

        # read labels
        self.labels = []
        with open(self.label_path, 'rb') as f1:
            _, size = struct.unpack('>II', f1.read(8))
            self.labels = array("B", f1.read())

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

    def process_images(self, images):
        """
        Processes images to create float tensor whose values lie between 0 & 1.
        An Image is converted to a float tensor, scaled and the values are sampled
        from a bernoulli dist, so that all values in image tensor are 0 or 1.
        """

        bernoulli_param = torch.tensor(images, dtype=torch.float) / 255.0
        x = torch.bernoulli(bernoulli_param)
        return x

    def __len__(self):
        return len(self.processed_image_tensor)
    
    def __getitem__(self, index):
        return self.processed_image_tensor[index]
    

class VAE(nn.Module):
    """Implements vanilla VAE."""

    def __init__(self, config: VAEConfig):
        super().__init__()

        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.rec_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x):
        """Runs both encoder and decoder and returns nelbo loss."""
        mu, var = self.encoder(x)
        epsilon_mean = torch.zeros_like(mu)
        epsilon_var = torch.ones_like(var)
        z = mu + torch.normal(epsilon_mean, epsilon_var) * torch.sqrt(var)

        bernoulli_logits = self.decoder(z)
        rec = self.rec_loss(input=bernoulli_logits, target=x).sum(-1)

        # simplified KL term 
        kl_t = 0.5 * (-torch.log(var) - 1 + var + (mu * mu))
        kl = torch.sum(kl_t, dim=-1)

        nelbo = rec + (self.config.kl_penalty * kl)

        nelbo = torch.mean(nelbo, dim=0)
        rec = torch.mean(rec, dim=0)
        kl = torch.mean(kl, dim=0)
        return nelbo, rec, kl
    
    def inference_forward(self, batch_size):
        """Samples z and only runs the decoder to generate image."""

        z = torch.normal(mean=torch.zeros(batch_size, self.config.z_dim), 
                         std=torch.ones(batch_size, self.config.z_dim))
        bernoulli_params = F.sigmoid(self.decoder(z))
        x = torch.bernoulli(bernoulli_params)
        return x


def train(vae_model: nn.Module, num_epochs, img_path, label_path):
        mnist_dataset = MNISTDataset(img_path, label_path)
        mnist_dataloader = DataLoader(mnist_dataset, batch_size=4)
        optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
        print('num training examples: ', len(mnist_dataset))

        for epoch in range(num_epochs):
            for batch, img in enumerate(mnist_dataloader):
                optimizer.zero_grad()
                loss, rec, kl = vae_model(img)
                loss.backward()
                optimizer.step()

                if batch % 10 == 0:
                    print(f'epoch: {epoch}, batch: {batch} | loss: {loss.item()}, rec: {rec.item()}, kl: {kl.item()}')


def sample(model: VAE):
    """Samples images from VAE model and creates a matrix plot of sampled images. """

    x = model.inference_forward(batch_size=200)
    images = x.reshape(200, 28, 28)
    figs, axs = plt.subplots(nrows=10, ncols=20, figsize=(28,28))

    idx = 0
    for ax in axs.reshape(-1):
        ax.imshow(images[idx].detach().numpy())
        idx += 1

    plt.tight_layout()
    plt.savefig('trained_vae_samples.png')
    plt.show()


def main():
    config = VAEConfig(
        z_dim=10,
        intermediate_dim=300,
        image_size=784,
        kl_penalty=1
    )
    model = VAE(config)
    img_path = '/kaggle/input/mnist-dataset/train-images.idx3-ubyte'
    label_path = '/kaggle/input/mnist-dataset/train-labels.idx1-ubyte'
    num_epochs = 1

    print('starting training !')
    train(model, num_epochs, img_path, label_path)
    print('training complete !')

    sample(model)
    torch.save(model.state_dict(), 'trained_vae_model.pt')


if __name__ == "__main__":
    main()