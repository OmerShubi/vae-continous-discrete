import numpy as np
import torch

from model import VariationalAutoencoder, DiscreteVAE
from utils import plot_latent, plot_reconstructed, interpolate_gif, image_grid_gif
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
from tqdm.auto import trange

torch.manual_seed(0)
plt.rcParams['figure.dpi'] = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_cont(vae, data, epochs=20):
    opt = torch.optim.Adam(vae.parameters(), lr=0.001)
    for epoch in trange(epochs):
        for x, y in tqdm(data):
            x = x.to(device)  # GPU
            opt.zero_grad()
            x_hat = vae(x)
            loss = F.binary_cross_entropy(x_hat, x, reduction='sum') + vae.encoder.kl
            loss.backward()
            opt.step()
    return vae


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') / x.shape[0]

    log_ratio = torch.log(qy * qy.size(-1) + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD


def train_disc(model, optimizer, data_loader, num_epochs=20, temp=1.0, hard=False):
    temp_min = 0.5
    ANNEAL_RATE = 0.00003
    model.train()
    train_loss = 0
    for epoch in trange(num_epochs):
        for batch_idx, (x, _) in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            x = x.to(device)  # GPU
            x_hat, qy = model(x, temp, hard)
            loss = loss_function(x_hat, x, qy)
            loss.backward()
            train_loss += loss.item() * len(x)
            optimizer.step()
            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)





def main():
    z_dim = 2

    data_loader = DataLoader(torchvision.datasets.CelebA('./data/celebA', transform=torchvision.transforms.ToTensor(), download=True),
                             batch_size=128,
                             shuffle=True)

    vae_cont = VariationalAutoencoder(z_dim).to(device)  # GPU
    vae_cont = train_cont(vae_cont, data_loader, epochs=2)

    # Viz
    plot_latent(vae_cont, data_loader)
    plot_reconstructed(vae_cont, r0=(-3, 3), r1=(-3, 3))
    x, y = data_loader.__iter__().next()  # hack to grab a batch
    x_1 = x[y == 1][1].to(device)  # find a 1
    x_2 = x[y == 3][1].to(device)  # find a 2
    interpolate_gif(vae_cont, "vae_cont", x_1, x_2)

    # Discrete

    N = 3
    K = 20  # one-of-K vector

    temp = 1.0
    hard = False

    vae_disc = DiscreteVAE(N, K).to(device)

    optimizer = torch.optim.Adam(vae_disc.parameters(), lr=1e-3)
    train_disc(vae_disc, optimizer, data_loader, 2, temp, hard)

    image_grid_gif(vae_disc,N,K)

if __name__ == '__main__':
    main()
