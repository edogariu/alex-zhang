from typing import Dict, Union, Tuple, NamedTuple
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelBase
import losses

"""
This is meant for the setup with learned appearance codes per images used in NeRF-W and BlockNeRF
We apply an overall reconstructive loss on RGB
"""

class AppearanceCodes(ModelBase):
    """
    a class to handle appearance code embeddings
    """
    def __init__(self, 
                 models: Union[nn.Module, Dict[str, nn.Module]],
                 loss_weights: Dict[str, float] = {},
                 model_name = 'Appearance + Structure Codes',
                 **kwargs):
        
        assert {'CODE_encoder', 'RGB_decoder'} <= models.keys(), 'must have the necessary encoders and decoders'
        super(AppearanceCodes, self).__init__(models, model_name, **kwargs)
        
        self._CODE_encoder = models['CODE_encoder']
        self._RGB_decoder = models['RGB_decoder']
        
        assert self._CODE_encoder.latent_dim + self._CODE_encoder.latent_dim == self._RGB_decoder.latent_dim, 'inconsistent latent dims for RGB reconstruction'
        
        self.appearance_latent_dim = self._CODE_encoder.latent_dim
        self.structure_latent_dim = self._CODE_encoder.latent_dim
        
        self._embedding = namedtuple('Embedding', ['CODE_appearance', 'CODE_structure'])
        
        if len(loss_weights) == 0:
            print('MODEL INFO: no loss weights were provided. i cant calculate the loss but i will do the best i can :)')
        else:
            print('MODEL INFO: loss weights dict:', loss_weights)
        self._loss_weights = loss_weights
        
    def infer(self, 
              x: torch.Tensor,
              idxs: torch.Tensor) -> NamedTuple:
        """
        embeds the stuff

        Parameters
        ----------
        x : torch.Tensor
            input batch of shape `(B, 4, H, W)`
            
        Returns
        -------
        NamedTuple[torch.Tensor]
            contains `RGB_appearance`, `RGB_structure` as keys
        """
        with torch.no_grad():
            rgb, depth = self._split_data(x)
            
            rgb_appearance, rgb_structure = self._CODE_encoder(idxs)
            return self._embedding(rgb_appearance, rgb_structure)
    
    def loss(self, 
             x: torch.Tensor, 
             idxs: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Loss for a batch of training examples. 
        This is where we choose to define the loss, which is what the training process attempts to minimize. This must be differentiable.

        Parameters
        ----------
        x : torch.Tensor
            input batch of shape `(B, 4, H, W)`
        # kl_weight : float
        #     weight for kl divergence loss from VAE encoding
        # contrastive_weight : float
        #     weight for contrastive loss between the structure embeddings
        # anticontrastive_weight : float
        #     weight for anticontrastive loss between the appearance and structure embeddings
            
        Returns
        ------------
        torch.Tensor
            compound loss
        float
            error
        """
        assert len(self._loss_weights) != 0, 'can\'t calculate loss without weights!'

        rgb, depth = self._split_data(x)
        
        # embed
        rgb_appearance, rgb_structure = self._CODE_encoder(idxs)
        
        # rgb reconstruction loss
        reconstruction = self._RGB_decoder(torch.cat((rgb_appearance, rgb_structure), dim=-1))
        reconstruction_loss = losses.reconstruction_loss(reconstruction, rgb)

        loss = reconstruction_loss

        return loss, reconstruction_loss.item()
    
    def eval_err(self, 
                 x: torch.Tensor, 
                 idxs: torch.Tensor,
                 **kwargs) -> Tuple[float]:
        """
        Error for a batch of training examples. 
        This is where we choose to define the error, which is what we want the model to ultimately do best at. This doesn't need to be differentiable.

        Parameters
        ----------
        x : torch.Tensor
            input batch of shape `(B, 4, H, W)`
            
        Returns
        ------------
        float
            error
        float
            loss
        """
        assert len(self._loss_weights) != 0, 'can\'t calculate loss without weights!'
        
        with torch.no_grad():
            loss, error = self.loss(x, idxs, **kwargs)
            return error, loss.item()
    
    def _split_data(self,
                    x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        splits data batch into the RGB and DEPTH modes. if we were to transition to 3D, here should be the only place in this file we would have to change

        Parameters
        ----------
        x : tuple of 2 torch.Tensor
            input batch of shape `(B, 4, H, W), (B, 1)`

        Returns
        -------
        _type_
            _description_
        """
        rgb, depth = torch.split(x, 3, dim=1)
        return rgb, depth
    