from datasets import RGBDDataset, RadiateDataset
from vae import Encoder, Decoder, EncoderEmbedding
from ensemble import Ensemble
from appearance_codes import AppearanceCodes
from trainer import Trainer
from utils import DEVICE

import torch
import numpy as np
import random
import argparse

def main(args):
    # ----------------------------------------------------------------------------------------------
    # --------------------------------- HYPERPARAMETERS --------------------------------------------
    # ----------------------------------------------------------------------------------------------
        
    resolution = (128, 128)
    dataset = RadiateDataset(resolution, n_data=9000, use_multiprocessing=True)
    
    model_args = {'code_latent_dim': 128,
                  'appearance_latent_dim': 128,
                  'structure_latent_dim': 128,
                  'hidden_dims': [32, 64, 128, 256, 512]}
    
    trainer_args = {'initial_lr': 0.0012, # 3e-5
                'lr_decay_period': 50,
                'lr_decay_gamma': 0.8,
                'weight_decay': 1e-5}
    
    train_args = {'num_epochs': 100,
                  'eval_every': 3,
                  'patience': 5,
                  'num_tries': 20}    
    
    loss_weights = {'kl_weight': 0.00005,  # 0.00005
                    'contrastive_weight': 0.02,  # 0.02
                    'anticontrastive_weight': 0.0005}  # 0.0005
    
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    
    models = {}
    
    # make ensemble
    if args.model_type == 'ensemble':
        multi_load = False
        dataset_args = {'batch_size': 64}
        
        models = {'RGB_appearance': Encoder(input_shape=[3, *resolution], latent_dim=model_args['appearance_latent_dim'], hidden_dims=model_args['hidden_dims']), 
                'RGB_structure': Encoder(input_shape=[3, *resolution], latent_dim=model_args['structure_latent_dim'], hidden_dims=model_args['hidden_dims']), 
                'DEPTH_structure': Encoder(input_shape=[1, *resolution], latent_dim=model_args['structure_latent_dim'], hidden_dims=model_args['hidden_dims'])}
        models['RGB_decoder'] = Decoder(input_shape=[3, *resolution], 
                                        latent_dim=model_args['appearance_latent_dim'] + model_args['structure_latent_dim'], 
                                        encoder_conv_out_shape=models['RGB_appearance'].conv_out_shape, 
                                        hidden_dims=model_args['hidden_dims'])
        model = Ensemble(models, loss_weights=loss_weights).float().to(DEVICE)
    elif args.model_type == 'code':
        multi_load = True
        dataset_args = {'batch_size': 64, 
                        'include_indices': True}
        
        models['CODE_encoder'] = EncoderEmbedding(latent_dim=model_args['code_latent_dim'])
        models['RGB_decoder'] = Decoder(input_shape=[3, *resolution], 
                                        latent_dim=model_args['code_latent_dim'] + model_args['code_latent_dim'], 
                                        hidden_dims=model_args['hidden_dims'])
        model = AppearanceCodes(models, loss_weights=loss_weights).float().to(DEVICE)
    
    # make dataloaders
    train_dataloader = dataset.get_dataloader('train', **dataset_args)
    val_dataloader = dataset.get_dataloader('val', **dataset_args)
    test_dataloader = dataset.get_dataloader('test', **dataset_args)
    
    # make trainer -- this assumes we want the same lr stuff for each model (but we can customize for each one)
    initial_lr = {k: trainer_args['initial_lr'] for k in models.keys()}
    lr_decay_period = {k: trainer_args['lr_decay_period'] for k in models.keys()}
    lr_decay_gamma = {k: trainer_args['lr_decay_gamma'] for k in models.keys()}
    weight_decay = {k: trainer_args['weight_decay'] for k in models.keys()}

    t = Trainer(model, train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay, multi_load=multi_load)
    
    # train!
    t.train(**train_args)
    
    
if __name__ == '__main__':
    print('Default Device:', DEVICE)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default="69420",
    )
    
    parser.add_argument(
        '--model_type', 
        type=str, default='ensemble', 
        choices=['ensemble','code'],
        help='type of model to use (default: ensemble)')

    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    main(args)