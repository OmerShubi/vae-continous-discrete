import torch
from model_utils import gumbel_softmax
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self, latent_dims, hidden_dim=512, input_dim=784):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.to_mean_logvar = nn.Linear(hidden_dim, 2 * latent_dims)

    def reparametrization_trick(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu, log_var = torch.split(self.to_mean_logvar(x), 2, dim=-1)
        self.kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return self.reparametrization_trick(mu, log_var)


class Decoder(nn.Module):
    def __init__(self, latent_dims, hidden_dim=512, output_dim=784):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 3, 28, 28))


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, hidden_dim=512, input_dim=2352, output_dim=2352):
        super().__init__()
        self.encoder = Encoder(latent_dims, hidden_dim=hidden_dim, input_dim=input_dim)
        self.decoder = Decoder(latent_dims, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class DiscreteVAE(nn.Module):
    def __init__(self, latent_dim, categorical_dim, input_dim=2352, output_dim=2352):
        super(DiscreteVAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.N = latent_dim
        self.K = categorical_dim

    def encoder(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decoder(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x, temp, hard):
        q = self.encoder(x.view(-1, self.input_dim))
        q_y = q.view(q.size(0), self.N, self.K)
        z = gumbel_softmax(q_y, temp, hard)
        return self.decoder(z), F.softmax(q_y, dim=-1).reshape(q.size(0) * self.N, self.K)


class JointVAE(nn.Module):
    def __init__(self, latent_dim_disc, latent_dim_cont, categorical_dim, input_dim=2352, output_dim=2352):
        super(JointVAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Encode Continous
        hidden_dim = 512
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.to_mean_logvar = nn.Linear(hidden_dim, 2 * latent_dim_cont)

        # Encode Disc
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim_disc * categorical_dim)

        # Decode
        decoder_input_dim = latent_dim_disc * categorical_dim + latent_dim_cont
        self.fc4 = nn.Linear(decoder_input_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.N = latent_dim_disc
        self.K = categorical_dim

    @staticmethod
    def reparametrization_trick(mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var / 2) * eps

    def encoder_cont(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu, log_var = torch.split(self.to_mean_logvar(x), 2, dim=-1)
        self.kl_cont = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # TODO mean?
        return self.reparametrization_trick(mu, log_var)

    def encoder_disc(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decoder(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x, temp, hard):
        z_cont = self.encoder_cont(x)

        q = self.encoder_disc(x.view(-1, self.input_dim))
        q_y = q.view(q.size(0), self.N, self.K)
        z_disc = gumbel_softmax(q_y, temp, hard)
        z = torch.cat([z_cont, z_disc], dim=1)

        return self.decoder(z), F.softmax(q_y, dim=-1).reshape(q.size(0) * self.N, self.K)
