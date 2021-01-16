import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import torch.utils
from tqdm import trange, tqdm

plt.rcParams['figure.dpi'] = 200

torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        # 2. makes the gradient equal to y_soft gradient
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y.reshape(bs, N * K)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy, output_dim):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, output_dim), reduction='sum')  # / x.shape[0]

    log_ratio = torch.log(qy * qy.size(-1) + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).sum()

    return BCE + KLD


def train_joint(model, optimizer, data_loader, save_path, num_epochs=20, temp=1.0, hard=False):
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
    torch.save(model, os.path.join(save_path, 'model.pth'))
