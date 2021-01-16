import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import VariationalAutoencoder, DiscreteVAE, JointVAE
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


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy, output_dim):

    BCE = F.binary_cross_entropy(recon_x, x.view(-1, output_dim), reduction='sum')# / x.shape[0]

    log_ratio = torch.log(qy * qy.size(-1) + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).sum()

    return BCE + KLD


def train_joint(model, optimizer, data_loader, num_epochs=20, temp=1.0, hard=False):
    temp_min = 0.5
    ANNEAL_RATE = 0.00003
    model.train()
    train_loss = 0
    for epoch in trange(num_epochs):
        print(f"epoch {epoch}")
        epoch_loss = 0
        for batch_idx, (x, _) in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            x = x.to(device)  # GPU
            x_hat, qy = model(x, temp, hard)
            loss = loss_function(x_hat, x, qy, model.output_dim)
            loss += model.kl_cont
            epoch_loss += loss
            loss.backward()
            train_loss += loss.item() * len(x)
            optimizer.step()
            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        print(f"epoch {epoch}, loss:{epoch_loss / len(data_loader.dataset)}")

def main():
    DEBUG = False
    z_dim = 2
    image_size = 64
    N = 3
    K = 20  # one-of-K vector
    IMAGE_PATH = './data'

    temp = 1.0
    hard = False
    num_workers = 0 if DEBUG else 5
    num_epochs = 1 if DEBUG else 5

    image_dim = image_size*image_size*3
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(IMAGE_PATH, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=1024, shuffle=True, num_workers=num_workers, drop_last=True)

    vae_joint = JointVAE(latent_dim_disc=N, latent_dim_cont=z_dim, categorical_dim=K, input_dim=image_dim, output_dim=image_dim).to(device)

    optimizer = torch.optim.Adam(vae_joint.parameters(), lr=1e-3)
    train_joint(vae_joint, optimizer=optimizer, data_loader=data_loader, num_epochs=num_epochs, temp=temp, hard=hard)

    # Viz
    # plot_latent(vae_joint, data_loader)
    plot_reconstructed(vae_joint, r0=(-3, 3), r1=(-3, 3))
    # interpolate_gif(vae_joint, "vae_cont", x_1, x_2)
    image_grid_gif(vae_joint, N, K, image_size)


if __name__ == '__main__':
    main()
