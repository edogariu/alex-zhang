from datasets import RGBDDataset
from vae import Encoder, Decoder
from ensemble import Ensemble
from trainer import Trainer
from utils import DEVICE

if __name__ == '__main__':
    
    # ----------------------------------------------------------------------------------------------
    # --------------------------------- HYPERPARAMETERS --------------------------------------------
    # ----------------------------------------------------------------------------------------------
    
    resolution = (128, 128)
    
    model_args = {'appearance_latent_dim': 128,
                  'structure_latent_dim': 128,
                  'hidden_dims': [32, 64, 128, 256, 512]}
    
    dataset_args = {'batch_size': 64}
    
    trainer_args = {'initial_lr': 0.001,
                    'lr_decay_period': 10,
                    'lr_decay_gamma': 0.7,
                    'weight_decay': 0.}
    
    train_args = {'num_epochs': 400,
                  'eval_every': 2,
                  'patience': 5,
                  'num_tries': 20}    
    
    loss_weights = {'kl_weight': 0.00025,
                    'contrastive_weight': 0.,
                    'anticontrastive_weight': 0.}
    
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    
    # make dataset
    dataset = RGBDDataset(resolution)
    train_dataloader = dataset.get_dataloader('train', **dataset_args)
    val_dataloader = dataset.get_dataloader('val', **dataset_args)
    test_dataloader = dataset.get_dataloader('test', **dataset_args)
    
    # make ensemble
    models = {'RGB_appearance': Encoder(input_shape=[3, *resolution], latent_dim=model_args['appearance_latent_dim'], hidden_dims=model_args['hidden_dims']), 
              'RGB_structure': Encoder(input_shape=[3, *resolution], latent_dim=model_args['structure_latent_dim'], hidden_dims=model_args['hidden_dims']), 
              'DEPTH_structure': Encoder(input_shape=[1, *resolution], latent_dim=model_args['structure_latent_dim'], hidden_dims=model_args['hidden_dims'])}
    models['RGB_decoder'] = Decoder(input_shape=[3, *resolution], 
                                    latent_dim=model_args['appearance_latent_dim'] + model_args['structure_latent_dim'], 
                                    encoder_conv_out_shape=models['RGB_appearance'].conv_out_shape, 
                                    hidden_dims=model_args['hidden_dims'])
    model = Ensemble(models, loss_weights=loss_weights).float().to(DEVICE)
    
    # make trainer -- this assumes we want the same lr stuff for each model (but we can customize for each one)
    initial_lr = {k: trainer_args['initial_lr'] for k in models.keys()}
    lr_decay_period = {k: trainer_args['lr_decay_period'] for k in models.keys()}
    lr_decay_gamma = {k: trainer_args['lr_decay_gamma'] for k in models.keys()}
    weight_decay = {k: trainer_args['weight_decay'] for k in models.keys()}

    t = Trainer(model, train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
    
    # train!
    t.train(**train_args)
    