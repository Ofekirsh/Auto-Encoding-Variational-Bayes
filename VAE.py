import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, latent_dim,device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )

    def sample(self, sample_size, mu=None, logvar=None):
        """
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        """
        if mu==None:
            mu = torch.zeros((sample_size,self.latent_dim)).to(self.device)

        if logvar == None:
            logvar = torch.zeros((sample_size,self.latent_dim)).to(self.device)


        z = self.z_sample(mu, logvar)
        z = self.upsample(z)
        z = z.reshape(-1, 64, 7, 7)
        p = self.decoder(z)
        return p

    def z_sample(self, mu, logvar):
        sample_size = (mu.shape[0], self.latent_dim)
        epsilon = torch.randn(sample_size).to(self.device)
        z = mu + epsilon * torch.exp(logvar / 2)
        return z

    def loss(self, x, recon, mu, logvar):
        """
        loss = I + II
        I - 1/n * sigma from i=0 until size of batch (BCE(x, recon))
        II - 1/2 * sigma from j=0 until size of features (1 + logvar - mu^2 - exp(logvar)
        """

        x = x.reshape(x.shape[0], -1)
        recon = recon.reshape(recon.shape[0], -1)

        # Reconstruction loss
        part1 = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)

        # KL Divergence
        part2 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        part2 = torch.mean(part2)

        loss = part1 + part2
        return loss

    def forward(self, x):
        enc = self.encoder(x)
        enc = enc.reshape(enc.shape[0], -1)
        mu_enc = self.mu(enc)
        logvar_enc = self.logvar(enc)
        z = self.z_sample(mu_enc, logvar_enc)
        z = self.upsample(z)
        z = z.reshape(-1, 64, 7, 7)
        p = self.decoder(z)
        return p, mu_enc, logvar_enc