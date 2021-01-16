import torch;

from model import VariationalAutoencoder
from utils import plot_latent, plot_reconstructed

torch.manual_seed(0)
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(vae, data, epochs=20):
    opt = torch.optim.Adam(vae.parameters(), lr = 0.001)
    for epoch in range(epochs):
        for x, y in tqdm(data):
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = vae(x)
            loss = F.binary_cross_entropy(x_hat, x, reduction='sum') + vae.encoder.kl
            loss.backward()
            opt.step()
    return vae


def main():
    z_dim = 2

    data_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data',
                                                                         transform=torchvision.transforms.ToTensor(), download=True),
                                              batch_size=128,
                                              shuffle=True)

    vae = VariationalAutoencoder(z_dim).to(device)  # GPU
    vae = train(vae, data_loader, epochs=1)
    plot_latent(vae, data_loader)

    plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))

if __name__ == '__main__':
    main()


