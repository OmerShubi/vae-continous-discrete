import torch;
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import torch.nn.functional as F
from PIL import Image
import IPython
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_latent(model, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = model.encoder_cont(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = []
    for i, z2 in enumerate(np.linspace(r1[1], r1[0], n)):
        for j, z1 in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[z1, z2]]).to(device)
            x_hat = autoencoder.decoder(z)
            img.append(x_hat)

    img = torch.cat(img)
    img = torchvision.utils.make_grid(img, nrow=12).permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(img, extent=[*r0, *r1])
    plt.show()


def sample_gumbel(shape, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-20):
    dims = len(logits.size())
    gumbel_noise = sample_gumbel(logits.size(), eps=eps).to(device)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    bs, N, K = logits.size()
    y_soft = gumbel_softmax_sample(logits.view(bs * N, K), tau=tau, eps=eps)

    if hard:
        k = torch.argmax(y_soft, dim=-1)
        y_hard = F.one_hot(k, num_classes=K)

        # 1. makes the output value exactly one-hot
        # 2.makes the gradient equal to y_soft gradient
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y.reshape(bs, N * K)


def interpolate_gif(autoencoder, filename, x_1, x_2, n=100):
    z_1 = autoencoder.encoder(x_1.unsqueeze(0))
    z_2 = autoencoder.encoder(x_2.unsqueeze(0))

    z = torch.stack([z_1 + (z_2 - z_1) * t for t in np.linspace(0, 1, n)])

    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy() * 255

    images_list = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1]  # loop back beginning

    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)

def image_grid_gif(model,N,K, image_size):
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
        z_cont = torch.randn(2).repeat(400, 1)
        z = torch.cat([z_cont, z_disc], dim=1).to(device)
        reconst_images = model.decoder(z)
        reconst_images = reconst_images.view(reconst_images.size(0), 3, image_size, image_size).detach().cpu()
        grid_img = torchvision.utils.make_grid(reconst_images, nrow=K).permute(1, 2, 0).numpy() * 255
        grid_img = grid_img.astype(np.uint8)
        images_list.append(Image.fromarray(grid_img))

    images_list[0].save(
        'dvae.gif',
        save_all=True,
        duration=700,
        append_images=images_list[1:],
        loop=10)

    IPython.display.IFrame("dvae.gif", width=900, height=450)


