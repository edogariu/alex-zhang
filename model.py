from abc import abstractmethod
import os
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.optim as optim

from utils import count_parameters, TOP_DIR_NAME

OPTIMIZER = optim.Adam

"""
This class might look real useless right about now. But I promise its practicing good design and maintainability practices!
I made it in order to allow for seamless addition of multiple model ensembles. For example, frameworks that use joint embeddings, or perhaps different models for different days. 
In general, this defines the API for how to interact with the models, and makes it easy to change it to add more complicated things. 
Without changing any method signatures or anything, all one has to do is reimplement `.infer()`, `.loss()`, and `.eval_err()`   :)
For an example of this, look at `jepa.py`
"""

class ModelBase():
    """
    a class to interact with any framework of models
    """
    def __init__(self, 
                 models: Union[nn.Module, Dict[str, nn.Module]],
                 model_name: str=None,
                 checkpoint_folder: str=os.path.join(TOP_DIR_NAME, 'checkpoints')):
        
        self._model_name = model_name if model_name else 'main'  # only used in the __str__() method and as the dict key if training a single nn.Module
        self._models = models if type(models) == dict else {self._model_name: models}

        self._optimizers = {}
        self._lr_schedulers = {}
        
        self._checkpoint_folder = checkpoint_folder

        if not os.path.isdir(self._checkpoint_folder):
            print('No existing folder at {}! I\'m gonna go ahead and make one :)'.format(self._checkpoint_folder))
            os.mkdir(self._checkpoint_folder)
            os.mkdir(os.path.join(self._checkpoint_folder, 'models'))
            os.mkdir(os.path.join(self._checkpoint_folder, 'optimizers'))

        
    def train(self):
        """
        Change models to train mode
        """
        for model in self._models.values():
            model.train()
        return self
        
    def eval(self):
        """
        Change models to evaluation mode
        """
        for model in self._models.values():
            model.eval()
        return self
    
    def float(self):
        """
        Change models to dtype float
        """
        for model in self._models.values():
            model.float()
        return self
        
    def to(self, device: torch.device):
        """
        Move models to given device.
        Parameters
        ----------
        device : torch.device
            device
        """
        for model in self._models.values():
            model.to(device)
        return self
        
    def init_optimizer_and_lr_scheduler(self, 
                                        initial_lr: Union[float, Dict[str, float]], 
                                        lr_decay_period: Union[int, Dict[str, int]], 
                                        lr_decay_gamma: Union[float, Dict[str, float]], 
                                        weight_decay: Union[float, Dict[str, float]]):
        """
        Creates optimiziers and learning rate schedulers for each model.
        Parameters
        ----------
        initial_lr : Union[float, Dict[str, float]]
            learning rate to start with for each model
        lr_decay_period : Union[int, Dict[str, int]]
            how many epochs between each decay step for each model
        lr_decay_gamma : Union[float, Dict[str, float]]
            size of each decay step for each model
        weight_decay : Union[float, Dict[str, float]]
            l2 regularization for each model
        """
        if type(initial_lr) == float: initial_lr = {k: initial_lr for k in self._models.keys()}
        if type(lr_decay_period) == int: lr_decay_period = {k: lr_decay_period for k in self._models.keys()}
        if type(lr_decay_gamma) == float: lr_decay_gamma = {k: lr_decay_gamma for k in self._models.keys()}
        if type(weight_decay) == float: weight_decay = {k: weight_decay for k in self._models.keys()}
        
        for k in self._models.keys():
            self._optimizers[k] = OPTIMIZER(self._models[k].parameters(), lr=initial_lr[k], weight_decay=weight_decay[k])
            self._lr_schedulers[k] = optim.lr_scheduler.StepLR(self._optimizers[k], step_size=lr_decay_period[k], gamma=lr_decay_gamma[k], verbose=True)
    
    def step_optimizers(self):
        """
        Steps optimizers for each model
        """
        for optimizer in self._optimizers.values():
            optimizer.step()
    
    def zero_grad(self):
        """
        Zeros gradients of optimizers for each model
        """
        for optimizer in self._optimizers.values():
            optimizer.zero_grad()
    
    def step_lr_schedulers(self):
        """
        Steps learning rate schedulers for each model
        """
        for lr_scheduler in self._lr_schedulers.values():
            lr_scheduler.step()
    
    @abstractmethod
    def infer(self):
        """
        Parameters
        ----------
        x : torch.Tensor
            input sequence of shape `(batch_size, in_dim)`
        Returns
        -------
        torch.Tensor
            output sequence of shape `(batch_size, out_dim)`
        """
        pass
    
    @abstractmethod
    def loss(self, 
             x):
        """
        Loss for a batch of training examples. 
        This is where we choose to define the loss, which is what the training process attempts to minimize. This must be differentiable.
        Parameters
        ----------
        x : torch.Tensor
            input sequence of shape `(batch_size, in_dim)`
        y : torch.Tensor
            target output sequence of shape `(batch_size, out_dim)`
            
        Returns
        ------------
        torch.Tensor
            loss
        float
            error
        """
        pass
    
    @abstractmethod
    def eval_err(self, 
                 x):
        """
        Error for a batch of training examples. 
        This is where we choose to define the error, which is what we want the model to ultimately do best at. This doesn't need to be differentiable.
        Parameters
        ----------
        x : torch.Tensor
            input sequence of shape `(batch_size, in_dim)`
        y : torch.Tensor
            target output sequence of shape `(batch_size, out_dim)`
            
        Returns
        ------------
        float
            error
        float
            loss
        """
        pass
    
    def __str__(self) -> str:
        s = '{} with the following parts:\n\n'.format(self._model_name)
        for k in self._models.keys():
            # s += str(self._models[k] + '\n')
            s += '\t{} with {} parameters\n'.format(k, count_parameters(self._models[k]))
        return s
    
    def load_checkpoint(self,
                        checkpoint_folder: str=None):
        if not checkpoint_folder:
            checkpoint_folder = self._checkpoint_folder
        print('MODEL INFO: loading checkpoint from {}'.format(checkpoint_folder))
        assert os.path.isdir(checkpoint_folder), '{} ain\'t a folder buddy'.format(checkpoint_folder)
        
        for k in self._models:
            model_filename = os.path.join(checkpoint_folder, 'models', '{}.pth'.format(k))
            opt_filename = os.path.join(checkpoint_folder, 'optimizers', '{}.pth'.format(k))
            
            assert os.path.isfile(model_filename)

            self._models[k].load_state_dict(torch.load(model_filename, map_location='cpu'), strict=True)
            if os.path.isfile(opt_filename):
                if k not in self._optimizers:
                    self._optimizers[k] = OPTIMIZER(self._models[k].parameters(), lr=0, weight_decay=0)
                self._optimizers[k].load_state_dict(torch.load(opt_filename, map_location='cpu'))
        return self
    
    def save_checkpoint(self,
                        checkpoint_folder: str=None):
        if not checkpoint_folder:
            checkpoint_folder = self._checkpoint_folder
        print('MODEL INFO: saving checkpoint to {}'.format(checkpoint_folder))
        assert os.path.isdir(checkpoint_folder), '{} ain\'t a folder buddy'.format(checkpoint_folder)
        
        for k in self._models:
            model_filename = os.path.join(checkpoint_folder, 'models', '{}.pth'.format(k))
            opt_filename = os.path.join(checkpoint_folder, 'optimizers', '{}.pth'.format(k))

            torch.save(self._models[k].state_dict(), model_filename)
            if k in self._optimizers:
                torch.save(self._optimizers[k].state_dict(), opt_filename)
        return self
    