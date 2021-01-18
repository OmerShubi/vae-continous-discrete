import os

import matplotlib.pyplot as plt
import torch
import torch.distributions
import torch.utils
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import JointVAE
from model_utils import train_joint
from vis_utils import plot_reconstructed, image_grid_gif, plot_latent
import datetime

torch.manual_seed(0)
plt.rcParams['figure.dpi'] = 200


# TODO
#  3. More visualizations (Use features? Combine cont and disc?)
#  6. CNN
#  7. Params
#  8. reproduce_hw3() - should be able to reproduce the results that you reported
#  9. Report


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DEBUG = False
    z_dim = 2
    image_size = 64
    N = 3
    K = 12  # one-of-K vector
    image_path = './data'
    base_path = './results'
    batch_size = 1024
    num_batches = 10
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    results_path = os.path.join(base_path, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    temp = 1.0
    hard = False
    num_workers = 0 if DEBUG else 5
    num_epochs = 1 if DEBUG else 10

    image_dim = image_size * image_size * 3

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(image_path, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    vae_joint = JointVAE(latent_dim_disc=N, latent_dim_cont=z_dim, categorical_dim=K, input_dim=image_dim, output_dim=image_dim).to(device)

    optimizer = torch.optim.Adam(vae_joint.parameters(), lr=1e-3)
    train_joint(model=vae_joint, optimizer=optimizer, data_loader=data_loader, save_path=results_path, num_epochs=num_epochs, temp=temp, hard=hard)

    # Viz
    plot_latent(vae_joint, data_loader, save_path=results_path, num_batches=num_batches)
    plot_reconstructed(vae_joint, r0=(-15, 15), r1=(-15, 15), n=6, N=N, K=K, image_size=image_size, save_path=results_path)
    # interpolate_gif(vae_joint, "vae_cont", x_1, x_2) # TODO fix function if want to use!
    image_grid_gif(vae_joint, N, K, image_size, save_path=results_path)


if __name__ == '__main__':
    main()
