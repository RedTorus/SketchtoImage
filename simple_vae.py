import torch
import torch.nn as nn
import torch.nn.functional as F


class myVAE(nn.Module):
    def __init__(self, latent_dim):
        super(myVAE, self).__init__()

        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.batchnorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.batchnorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.batchnorm2d(128),
            nn.ReLU()
        )
             
        self.mean_layer = nn.Linear(self.encode_size, latent_dim)  # input: (N,hidden_dim) -> output: (N, Z)
        self.logvar_layer = nn.Linear(self.encode_size, latent_dim)  # input: (N,hidden_dim) -> output: (N, Z)
        

        # input: (N, Z) -> output: (N,1,H,W)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.batchnorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.batchnorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )  
           
    def forward(self, x):
        encoded = self.encoder(x)
        
        encoded_size = encoded.size()
        encoded_size = torch.prod(torch.tensor(encoded_size))

        mu = nn.Linear(encoded_size, self.latent_dim)(encoded)
        logvar = nn.Linear(encoded_size, self.latent_dim)(encoded)

        z = self.reparametrize(mu, logvar)            
        x_hat = self.decoder(z)

        return x_hat, mu, logvar

    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def loss_function(self, x_hat, x, mu, logvar):
        KL_loss = -0.5 * torch.sum((1 + logvar - mu**2 - torch.exp(logvar)), 1)
        
        reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="none")
        reconstruction_loss = torch.sum(reconstruction_loss, dim=(1,2,3))
        loss = KL_loss + reconstruction_loss
        return loss