from typing import Dict, Union, Tuple, NamedTuple
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelBase
import losses

"""
This is meant for the setup with two encoders for RGB data, one encoder for depth data, and one decoder for RGB data
There are three latent spaces: RGB_appearance, RGB_structure, and DEPTH_structure

We apply a contrastive loss between RGB_structure and DEPTH_structure, an 'anti-contrastive loss' between RGB_appearance and RGB_structure (or DEPTH_structure), and an overall reconstructive loss on RGB
"""

class Ensemble(ModelBase):
    """
    a class to handle joint embeddings
    """
    def __init__(self, 
                 models: Union[nn.Module, Dict[str, nn.Module]],
                 loss_weights: Dict[str, float] = {},
                 model_name = 'Appearance + Structure Embedding Framework',
                 **kwargs):
        
        assert {'RGB_appearance', 'RGB_structure', 'DEPTH_structure', 'RGB_decoder'} <= models.keys(), 'must have the necessary encoders and decoders'
        super(Ensemble, self).__init__(models, model_name, **kwargs)
        
        self._RGB_appearance = models['RGB_appearance']
        self._RGB_structure = models['RGB_structure']
        self._DEPTH_structure = models['DEPTH_structure']
        self._RGB_decoder = models['RGB_decoder']
        
        assert self._RGB_appearance.latent_dim + self._RGB_structure.latent_dim == self._RGB_decoder.latent_dim, 'inconsistent latent dims for RGB reconstruction'
        assert self._RGB_structure.latent_dim == self._DEPTH_structure.latent_dim, 'inconsistent latent dims for RGB/DEPTH contrastive learning'
        assert self._RGB_appearance.latent_dim == self._DEPTH_structure.latent_dim, 'inconsistent latent dims for RGB/DEPTH anti-contrastive learning'
        
        self.appearance_latent_dim = self._RGB_appearance.latent_dim
        self.structure_latent_dim = self._RGB_structure.latent_dim
        
        self._embedding = namedtuple('Embedding', ['RGB_appearance', 'RGB_structure', 'DEPTH_structure'])
        
        if len(loss_weights) == 0:
            print('MODEL INFO: no loss weights were provided. i cant calculate the loss but i will do the best i can :)')
        else:
            print('MODEL INFO: loss weights dict:', loss_weights)
        self._loss_weights = loss_weights
        
    def infer(self, 
              x: torch.Tensor) -> NamedTuple:
        """
        embeds the stuff

        Parameters
        ----------
        x : torch.Tensor
            input batch of shape `(B, 4, H, W)`
            
        Returns
        -------
        NamedTuple[torch.Tensor]
            contains `RGB_appearance`, `RGB_structure`, `DEPTH_structure` as keys
        """
        with torch.no_grad():
            rgb, depth = self._split_data(x)
            
            rgb_appearance = self._RGB_appearance(rgb)
            rgb_structure = self._RGB_structure(rgb)
            depth_structure = self._DEPTH_structure(depth)
            return self._embedding(rgb_appearance, rgb_structure, depth_structure)
    
    def loss(self, 
             x: torch.Tensor) -> Tuple[torch.Tensor]:
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
        rgb_appearance = self._RGB_appearance(rgb)
        rgb_structure = self._RGB_structure(rgb)
        depth_structure = self._DEPTH_structure(depth)
        
        # rgb reconstruction loss
        reconstruction = self._RGB_decoder(torch.cat((rgb_appearance, rgb_structure), dim=-1))
        reconstruction_loss = losses.reconstruction_loss(reconstruction, rgb)
        
        # kl divergence loss
        kl_divergence_loss = self._RGB_appearance.kld_loss + self._RGB_structure.kld_loss + self._DEPTH_structure.kld_loss if self._loss_weights['kl_weight'] > 0 else torch.tensor(0).to(x.device)
        
        # contrastive losses
        contrastive_loss = losses.contrastive_loss(rgb_structure, depth_structure) if self._loss_weights['contrastive_weight'] > 0 else torch.tensor(0).to(x.device)
        anticontrastive_loss = losses.anticontrastive_loss(rgb_appearance, depth_structure) if self._loss_weights['anticontrastive_weight'] > 0 else torch.tensor(0).to(x.device)
        
        # print(reconstruction_loss.item(), 
        #       kl_divergence_loss.item() * self._loss_weights['kl_weight'],
        #       contrastive_loss.item() * self._loss_weights['contrastive_weight'], 
        #       anticontrastive_loss.item() * self._loss_weights['anticontrastive_weight'])

        loss = reconstruction_loss + \
               kl_divergence_loss * self._loss_weights['kl_weight'] + \
               contrastive_loss * self._loss_weights['contrastive_weight'] + \
               anticontrastive_loss * self._loss_weights['anticontrastive_weight']
        
        # print('kl: {}, con: {}, anticon: {}'.format(kl_divergence_loss, contrastive_loss, anticontrastive_loss))
        return loss, reconstruction_loss.item()
    
    def eval_err(self, 
                 x: torch.Tensor,
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
            loss, error = self.loss(x, **kwargs)
            return error, loss.item()
    
    def _split_data(self,
                    x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        splits data batch into the RGB and DEPTH modes. if we were to transition to 3D, here should be the only place in this file we would have to change

        Parameters
        ----------
        x : torch.Tensor
            input batch of shape `(B, 4, H, W)`

        Returns
        -------
        _type_
            _description_
        """
        rgb, depth = torch.split(x, 3, dim=1)
        return rgb, depth
    