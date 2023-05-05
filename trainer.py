from typing import Dict, Union
import os
import numpy as np
import tqdm

import torch
import torch.utils.data as D

from model import ModelBase, TOP_DIR_NAME

class Trainer():
    def __init__(self, 
                 model: ModelBase, 
                 train_dataloader: D.DataLoader,
                 val_dataloader: D.DataLoader,
                 initial_lr: Union[float, Dict[str, float]], 
                 lr_decay_period: Union[int, Dict[str, int]], 
                 lr_decay_gamma: Union[float, Dict[str, float]], 
                 weight_decay: Union[float, Dict[str, float]]):
        """
        Trainer object to train a model. Uses given optimizer, StepLR learning rate scheduler, and a patience algorithm.
        
        Parameters
        ------------
        model : Model
            single model or ensemble to train
        train_dataloader : D.DataLoader
            dataloader for training data
        val_dataloader : D.DataLoader
            dataloader for validation data. If this is `None`, we do not validate
        initial_lr : Union[float, Dict[str, float]]
            learning rate to start with for each model
        lr_decay_period : Union[int, Dict[str, int]]
            how many epochs between each decay step for each model
        lr_decay_gamma : Union[float, Dict[str, float]]
            size of each decay step for each model
        weight_decay : Union[float, Dict[str, float]]
            l2 regularization for each model
        """
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if val_dataloader is None:
            print('TRAINING INFO: No validation dataloader provided. Skipping validation.')

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('TRAINING INFO: Using {} for training'.format(self.device))

        # prep model and optimizer and scheduler
        self.model.train().to(self.device)
        if not self.model._optimizers: self.model.init_optimizer_and_lr_scheduler(initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)

        # prep statistics
        self.train_losses = {}
        self.val_losses = {}
        self.val_errors = {}
        self.best_val_loss = (0, float('inf'))  # epoch num and value of best validation loss

    def train_one_epoch(self, epoch_num: int):
        self.model.train()
        print()
        print('-------------------------------------------------------------')
        print('------------------  TRAIN - EPOCH NUM {}  -------------------'.format(epoch_num))
        print('-------------------------------------------------------------')
        
        avg_loss = 0.
        i = 0
        pbar = tqdm.tqdm(self.train_dataloader)
        for x, in pbar:
            x = x.to(self.device)
            self.model.zero_grad()
            loss = self.model.loss(x)
            loss.backward()
            self.model.step_optimizers()
            loss = loss.item()
            avg_loss += loss
            pbar.set_postfix({'batch loss': loss})
            i += 1
            
        avg_loss /= i
        self.train_losses[epoch_num] = avg_loss
        print('TRAINING INFO: avg batch training loss for epoch {}: {}'.format(epoch_num, round(avg_loss, 6)))
        self.model.step_lr_schedulers()
        pass

    def eval(self, epoch_num: int) -> float:
        self.model.eval()
        print()
        print('-------------------------------------------------------------')
        print('-------------------  VAL - EPOCH NUM {}  -------------------'.format(epoch_num))
        print('-------------------------------------------------------------')
        
        avg_err = 0.
        avg_loss = 0.
        i = 0
        with torch.no_grad():
            pbar = tqdm.tqdm(self.val_dataloader)
            for x, in pbar:
                x = x.to(self.device)
                err, loss = self.model.eval_err(x)
                avg_err += err
                avg_loss += loss
                pbar.set_postfix({'batch loss': loss})
                i += 1

        avg_err /= i
        avg_loss /= i
        self.val_errors[epoch_num] = avg_err
        self.val_losses[epoch_num] = avg_loss
        print('TRAINING INFO: avg validation error and batch loss for epoch {}: {}       {}'.format(epoch_num, round(avg_err, 6), round(avg_loss, 6)))
        return avg_err  # this is the metric we apply the patience algorithm on

    def train(self, 
              num_epochs: int, 
              eval_every: int, 
              patience: int, 
              num_tries: int):
        """
        Train the model. Applies patience -- every `eval_every` epochs, we eval on validation data using a metric that might not be the loss function. 
        We expect the model to improve in this metric as we train: if after `patience` validation steps the model has still not improved, we reset to the best previous checkpoint.
        We attempt this for `num_tries` attempts before we terminate training early.
        
        Parameters
        ---------------
        num_epochs : int
            number of epochs to train for
        eval_every : int
            interval (measured in epochs) between valiation evaluations
        patience : int
            number of evaluations without improvement before we reset to best checkpoint
        num_tries : int
            number of checkpoint resets before early stopping
        Returns 
        --------------
        model : Model
            the trained model or ensemble
        """
        print('TRAINING INFO: Training the following ensemble for {} epochs:\n{}'.format(num_epochs, self.model))
        patience_counter = 0
        tries_counter = 0
        for e in range(num_epochs):
            try:
                self.train_one_epoch(e)
                if e % eval_every == 0:
                    if self.val_dataloader is not None:
                        val_loss = self.eval(e)
                        if val_loss < self.best_val_loss[1]:  # measure whether our model is improving
                            self.best_val_loss = (e, val_loss)
                            patience_counter = 0
                            self.model.save_checkpoint()
                            print('TRAINING INFO: Saved checkpoint for epoch num {}'.format(e))
                        else:
                            patience_counter += 1
                            print('TRAINING INFO: Patience {} hit'.format(patience_counter))
                            if patience_counter >= patience:  # if our model has not improved after `patience` evaluations, reset to best checkpoint
                                tries_counter += 1
                                patience_counter = 0
                                self.model.load_checkpoint()
                                print('TRAINING INFO: Loaded checkpoint from epoch num {}'.format(self.best_val_loss[0]))
                                print('TRAINING INFO: Try {} hit'.format(tries_counter))
                                if tries_counter >= num_tries:  # if our model has reset to best checkpoint `num_tries` times, we are done
                                    print('TRAINING INFO: Stopping training!')
                                    break
                    else:
                        self.model.save_checkpoint()
                        print('TRAINING INFO: Saved checkpoint for epoch num {}'.format(e))
            except KeyboardInterrupt:
                print('Catching keyboard interrupt!!!')
                self.finish_up(e)
                return self.model
                # exit(0)
        self.finish_up(num_epochs)
        return self.model

    def finish_up(self, e: int):
        val_loss = self.eval(e) if self.val_dataloader else -float('inf')
        if val_loss < self.best_val_loss[1]:
            self.model.save_checkpoint()
            print('TRAINING INFO: Saved checkpoint at the end of training!')

        k = np.array(list(self.train_losses.keys()))
        v = np.array(list(self.train_losses.values()))
        np.save(os.path.join(TOP_DIR_NAME, + 'plots', 'train_losses.npy'), np.stack([k, v], axis=0))

        k = np.array(list(self.val_losses.keys()))
        v = np.array(list(self.val_losses.values()))
        np.save(os.path.join(TOP_DIR_NAME, + 'plots', 'val_losses.npy'), np.stack([k, v], axis=0))

        k = np.array(list(self.val_errors.keys()))
        v = np.array(list(self.val_errors.values()))
        np.save(os.path.join(TOP_DIR_NAME, + 'plots', 'val_errors.npy'), np.stack([k, v], axis=0))
        
        print('\nTRAINING INFO: Done :)')
        