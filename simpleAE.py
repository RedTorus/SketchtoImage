import torch
import torch.nn as nn

"""
Simple implementation of a convolutional variational autoencoder and an auto encoder.

It takes an images of size (N, C, H, W) and encodes them to size of (N, latent_dim, H/8, W/8).

Conv_out_size: number of channels in the output of the encoder
latent_dim: dimension of the channels in the latent space (only if VAE is used)
num_classes: number of classes in the dataset (if you want to do class conditioned encoding)
variation: if True, it will use a VAE, otherwise it will use a simple AE
device: cuda or cpu
in_channel: number of channels in the input image

"""
class myAE(nn.Module):
    def __init__(self, conv_out_size, latent_dim, num_classes = None, variation = False, device = "cuda", in_channel=3):
        super(myAE, self).__init__()
        self.variation = variation
        self.latent_dim = latent_dim
        self.conv_out_size = conv_out_size
        self.num_classes = num_classes
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=conv_out_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_out_size),
            nn.ReLU()
        )

        self.mu = nn.Conv2d(in_channels = conv_out_size, out_channels = latent_dim, kernel_size = 1, stride = 1, padding = 0)
        self.logvar = nn.Conv2d(in_channels = conv_out_size, out_channels = latent_dim, kernel_size = 1, stride = 1, padding = 0)

        if num_classes is not None:
            self.embed = nn.Embedding(num_classes, 3)

        # input: (N, Z) -> output: (N,1,H,W)
        if variation==True:
            inc=latent_dim
        else:
            inc=conv_out_size
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = inc, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        
            nn.ConvTranspose2d(in_channels=32, out_channels=in_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
   
    def forward(self, x, condition = None):
        if self.variation:                     
            encoded = self.encoder(x)
            

            mu = self.mu(encoded)
            logvar = self.logvar(encoded)

            z = self.reparameterize(mu, logvar)
            
            if condition is not None:
                #print("Condition shape",condition.shape)
                embed = self.embed(condition)
                #print("embedding shape",embed.shape)
                embed = embed.view(z.shape[0],z.shape[1],1,1)
                #print("embedding sape after trafo",embed.size())
                #print("z shape", z.size())
                z = z + embed
                          
            x_hat = self.decoder(z)
            return x_hat, mu, logvar
            
        else:
            #encoded = self.encoder(x)
            
            decoded = self.decoder(self.encoder(x))

            return decoded #encoded, decoded

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def loss_function(self, x_hat, x, mu, logvar):
            
        KL_loss = -0.5 * torch.sum((1 + logvar - mu**2 - torch.exp(logvar)), 1)
        KL_loss = torch.mean(KL_loss, dim = (1, 2))
        
        reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="none")
        reconstruction_loss = torch.mean(reconstruction_loss, dim=(1,2,3))
        
        loss = 0.0001 * KL_loss + 100*reconstruction_loss
        return loss
