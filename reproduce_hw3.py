import datetime
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from main import JointVAE
from vis_utils import plot_latent, image_grid_gif, plot_reconstructed, interpolate_gif


def reproduce_hw3():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_load_path = '/home/student/hw3/results/2021-01-16_20:49:03/model.pth'
    z_dim = 2
    image_size = 64
    N = 3
    K = 20  # one-of-K vector
    image_path = './data'
    base_path = './results'
    batch_size = 1024
    num_batches = 10
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    results_path = os.path.join(base_path, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    num_workers = 5

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), ])
    dataset = ImageFolder(image_path, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             drop_last=True)

    vae_joint_model = torch.load(model_load_path, map_location=lambda storage, loc: storage)
    vae_joint_model.to(device)
    # Viz
    # plot_latent(vae_joint_model, data_loader, save_path=results_path, num_batches=num_batches)
    # plot_reconstructed(vae_joint_model, r0=(-15, 15), r1=(-15, 15), n=6, N=N, K=K, image_size=image_size,
    #                    save_path=results_path)
    interpolate_gif(vae_joint_model, results_path,
                    z_0_low=-15, z_0_upper=15, z_1=12, N=N, K=K, image_size=image_size)  # TODO fix function if want to use!
    # image_grid_gif(vae_joint_model, N, K, image_size, save_path=results_path)


if __name__ == '__main__':
    reproduce_hw3()
