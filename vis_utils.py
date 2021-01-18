import os

import torch
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

plt.rcParams['figure.dpi'] = 200

torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def interpolate_gif(model, save_path, z_0_low, z_0_upper, z_1, N=3, K=20, image_size=128, n=100):
    images_list = []
    for t in np.linspace(0, 1, n):
        ind = torch.zeros(N, 1).long()
        ind[1] = 5
        ind[0] = 5
        ind[2] = 5
        z_disc = F.one_hot(ind, num_classes=K).squeeze(1).view(1, -1).float()

        z_cont_1 = z_0_low + (z_0_upper - z_0_low) * t
        z_cont = torch.Tensor([[z_cont_1, z_1]])

        z = torch.cat([z_cont, z_disc], dim=1).to(device)
        x_hat = model.decoder(z)
        reconst_image = x_hat.view(x_hat.size(0), 3, image_size, image_size).detach().cpu()
        grid_img = torchvision.utils.make_grid(reconst_image, nrow=1).permute(1, 2, 0).numpy() * 255
        grid_img = grid_img.astype(np.uint8)
        images_list.append(Image.fromarray(grid_img).resize((256,256)))

    images_list = images_list + images_list[::-1]  # loop back beginning

    images_list[0].save(
        os.path.join(save_path, 'cont.gif'),
        save_all=True,
        append_images=images_list[1:],
        loop=1)


def image_grid_gif(model, N, K, image_size, save_path):
    ind = torch.zeros(N, 1).long()
    images_list = []
    for k in range(K):
        to_generate = torch.zeros(K * K, N, K)
        index = 0
        for i in range(K):
            for j in range(K):
                ind[1] = k
                ind[0] = i
                ind[2] = j
                z = F.one_hot(ind, num_classes=K).squeeze(1)
                to_generate[index] = z
                index += 1

        z_disc = to_generate.view(-1, K * N)
        z_cont = torch.randn(2).repeat(K * K, 1)
        z = torch.cat([z_cont, z_disc], dim=1).to(device)
        reconst_images = model.decoder(z)
        reconst_images = reconst_images.view(reconst_images.size(0), 3, image_size, image_size).detach().cpu()
        grid_img = torchvision.utils.make_grid(reconst_images, nrow=K).permute(1, 2, 0).numpy() * 255
        grid_img = grid_img.astype(np.uint8)
        images_list.append(Image.fromarray(grid_img))

    images_list[0].save(
        os.path.join(save_path, 'disc.gif'),
        save_all=True,
        duration=700,
        append_images=images_list[1:],
        loop=10)


def plot_latent(model, data, save_path, num_batches=100):
    z0 = []
    z1 = []
    for i, (x, y) in enumerate(data):
        z = model.encoder_cont(x.to(device))
        z = z.to('cpu').detach().numpy()
        z0.extend(z[:, 0])
        z1.extend(z[:, 1])
        if i > num_batches:
            # plt.colorbar()
            break

    plt.scatter(z0, z1, c=[0] * len(z0))  # , c=y, cmap='tab10')
    plt.title("Continuous Latent Variables")
    plt.xlabel("$z_0$")
    plt.ylabel("$z_1$")
    plt.savefig(os.path.join(save_path, 'scatter_plot.png'))
    plt.show()


def plot_reconstructed(model, r0, r1, n, N, K, image_size, save_path):
    img = []

    ind = torch.zeros(N, 1).long()

    ind[1] = 5
    ind[0] = 5
    ind[2] = 5
    z_disc = F.one_hot(ind, num_classes=K).squeeze(1).view(1, -1).float()
    for i, z2 in enumerate(np.linspace(r1[1], r1[0], n)):
        for j, z1 in enumerate(np.linspace(*r0, n)):
            z_cont = torch.Tensor([[z1, z2]])

            z = torch.cat([z_cont, z_disc], dim=1).to(device)
            x_hat = model.decoder(z)
            reconst_image = x_hat.view(x_hat.size(0), 3, image_size, image_size).detach().cpu()
            img.append(reconst_image)

    img = torch.cat(img)
    img = torchvision.utils.make_grid(img, nrow=n).permute(1, 2, 0).numpy()
    plt.imshow(img, extent=[*r0, *r1])
    plt.title("Reconstructed Images, Continuous Variable Adjustment")
    plt.xlabel("$z_0$")
    plt.ylabel("$z_1$")
    plt.savefig(os.path.join(save_path, 'cont_plot.png'))

    plt.show()


def plot_loss(BCE_loss, KL_loss, save_path):
    plt.plot(list(range(1, len(BCE_loss) + 1)), BCE_loss, label='BCE loss')
    plt.plot(list(range(1, len(KL_loss) + 1)), KL_loss, label='KL loss')
    plt.title("BCE and KL Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.show()
