from typing import List, Tuple
import numpy as np

import torch
from torch import nn
import torchvision.transforms as T

from utils import DEVICE

"""
The following is HEAVILY INSPIRED by the implementations in https://github.com/AntixK/PyTorch-VAE/
(but i think i did some things better such as not restricting to 64x64 hehe)
"""

class Encoder(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int],
                 latent_dim: int,
                 hidden_dims: List=[32, 64, 128, 256, 512]):
        """
        creates VAE encoder

        Parameters
        ----------
        input_shape : Tuple[int]
            dimensions of image to encode, in `(C x H x W)` form
        latent_dim : int
            latent dimension to encode into
        hidden_dims : List, optional
            number of channels in each layer of convolutional body, by default `[32, 64, 128, 256, 512]`
        """
        super(Encoder, self).__init__()
        
        assert len(input_shape) == 3, 'images must have 3 dimensions'
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.device = DEVICE

        # build convolutional encoder
        modules = []
        in_channels = self.input_shape[0]
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 
                              out_channels=h_dim,
                              kernel_size=3, 
                              stride=2, 
                              padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        
        # store some shapes to use later
        with torch.no_grad():
            test = self.encoder(torch.zeros((1, *self.input_shape)))
            self.conv_out_shape = test.shape[1:]
            print('ENCODER INFO: after convolutions but before flattening, the shape is {}x{}x{}'.format(*self.conv_out_shape))
            conv_out_dim = np.prod(self.conv_out_shape)
            del test
        
        # MLPs to project to mean and log variance
        self.fc_mu = nn.Linear(conv_out_dim, latent_dim)
        self.fc_var = nn.Linear(conv_out_dim, latent_dim)
    
    def encode(self, 
               input: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def reparameterize(self, 
                       mu: torch.Tensor, 
                       logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) using N(0, 1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, 
                input: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(input)
        self.kld_loss = self.kl_divergence_loss(mu, log_var)
        z = self.reparameterize(mu, log_var)
        return z
    
    def kl_divergence_loss(self,
                           mu: torch.Tensor,
                           log_var: torch.Tensor) -> torch.Tensor:
        """
        measures KL divergence against gaussian
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        kld_loss = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=1).mean(dim=0)
        return kld_loss
        

class Decoder(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int],
                 latent_dim: int,
                 hidden_dims: List=[32, 64, 128, 256, 512],
                 encoder_conv_out_shape: List=None):
        """
        creates VAE decoder

        Parameters
        ----------
        input_shape : Tuple[int]
            dimensions of image to encode, in `(C x H x W)` form
        latent_dim : int
            latent dimension to encode into
        hidden_dims : List, optional
            number of channels in each layer of convolutional body, by default `[32, 64, 128, 256, 512]`
        encoder_conv_out_shape : List[int], optional
            shape of unflattened output of conv tower, by default `None`
        """
        super(Decoder, self).__init__()
        
        assert len(input_shape) == 3, 'images must have 3 dimensions'
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.conv_in_shape = encoder_conv_out_shape if encoder_conv_out_shape else [hidden_dims[-1], 2, 2]
        assert self.conv_in_shape[0] == hidden_dims[-1], 'dims don\'t match between encoder and decoder convolution'

        # build decoder input MLP
        self.decoder_input = nn.Sequential(
                                nn.Linear(latent_dim, latent_dim // 2),
                                nn.LeakyReLU(), 
                                nn.Linear(latent_dim // 2, np.prod(self.conv_in_shape)))

        # build convolutional decoder
        modules = []
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(self.hidden_dims[-1], 
                                      self.input_shape[0],
                                      kernel_size=3, 
                                      padding=1),
                            nn.Tanh())
        
        # do some dimension counting to make sure we aren't using resize unreasonably
        with torch.no_grad():
            test = torch.zeros((1, self.latent_dim))
            test = self.final_layer(self.decoder(self.decoder_input(test).view(-1, *self.conv_in_shape)))
            print('DECODER INFO: before resizing, generations have resolution {}x{}!'.format(*test.shape[2:]))
            del test
        
        self.resize = T.Resize(self.input_shape[1:], antialias=True)

    def forward(self, 
                z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        """
        result = self.decoder_input(z)
        result = result.view(-1, *self.conv_in_shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = self.resize(result)
        return result
    
if __name__ == '__main__':
    input_shape = (3, 128, 128)
    x = torch.zeros((5, *input_shape))
    enc = Encoder(input_shape, 128)
    dec = Decoder(input_shape, 128)
    emb = enc(x)
    rec = dec(emb)
    print(rec.shape)