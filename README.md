# alex-zhang

Hi alex. 

I coded up a whole bunch of stuff. Everything should actually be clean and polished, and the entrypoint is `run.py`. The hyperparameters of interest are there. I've tested it and it seems to work to train a VAE and do the right thing: take a look at the below example (original on left & reconstruction on right), which was trained on my mac in liek 10 minutes lol (its blurry, which might have something to do with me not turning on the KL divergence loss):

![image](https://user-images.githubusercontent.com/78564140/236430176-4a02ef47-0cd6-4202-9483-28380fd4a4b1.png)
![image](https://user-images.githubusercontent.com/78564140/236432497-d0af29a9-1e1c-4610-8f21-2f1718e09577.png)

### things i need to fix and/or next steps
- for some reason, whenever i set any of the loss weights (kl, contrastive, and anticontrastive) to something nonzero, loss eventually goes to nan. I think im calculating something wrong or dividing by something that goes to 0, idk ill fix it tomorrow
- gotta get the nuscenes dataset goin as well
- GOTTA ANALYZE DA LATENT SPACE. its kinda useless since i dont have the contrastive learning enabled (it's set to 0 in `run.py` and when i increase it the loss goes to nan, i must fix this), but still may be interesting. since the dataset im playing with only has like 14 scenes, it seems reasonable to expect good clustering

### things u need to do to set it up (i might write a batch script for this tmrw morning idc)
- `pip install requirements.txt` or whatever. im using a venv but u just gotta get those libraries somehow
- make the following folders inside the main directory
  - make a folder called `checkpoints/` and put two folders inside of it; one called `models/` and one called `optimizers/`
  - make a folder called `plots/`
  - make a folder called `data/`; inside of it, place the unzipped version of this file http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/rgbd-scenes-v2_imgs.zip, gotten from this link http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/
    - this is a RGB-D dataset of some random objects and scenes. i wanna make sure training and everything is stable on this first before doing stuff with more exotic datasets and especially before LIDAR
    - in `datasets.py` u can uncomment the thing to actually read through the dataset. it takes a while to load all the files, so i loaded them once and stored em in a `.npy` file to keep loading from. i recommend this :)
    - the images are in fairly high res. i downsampled to 64x64 for my lil experiment on the mac, but i think the full res is 640x480 or something
  


#### also if u want code to inference a thingy to reconstruct, here it is

    import matplotlib.pyplot as plt

    import torch

    from vae import Encoder, Decoder
    from datasets import RGBDDataset
    from ensemble import Ensemble
    from utils import DEVICE

    resolution = (64, 64)

    model_args = {'appearance_latent_dim': 64,
                  'structure_latent_dim': 64,
                  'hidden_dims': [32, 64, 128, 256]}

    dataset = RGBDDataset(resolution)
    dataloader = dataset.get_dataloader('all', 1)

    # make ensemble
    models = {'RGB_appearance': Encoder(input_shape=[3, *resolution], latent_dim=model_args['appearance_latent_dim'], hidden_dims=model_args['hidden_dims']), 
              'RGB_structure': Encoder(input_shape=[3, *resolution], latent_dim=model_args['structure_latent_dim'], hidden_dims=model_args['hidden_dims']), 
              'DEPTH_structure': Encoder(input_shape=[1, *resolution], latent_dim=model_args['structure_latent_dim'], hidden_dims=model_args['hidden_dims'])}
    models['RGB_decoder'] = Decoder(input_shape=[3, *resolution], 
                                    latent_dim=model_args['appearance_latent_dim'] + model_args['structure_latent_dim'], 
                                    encoder_conv_out_shape=models['RGB_appearance'].conv_out_shape, 
                                    hidden_dims=model_args['hidden_dims'])
    model = Ensemble(models, loss_weights=None).float().to(DEVICE)
    model.load_checkpoint()

    with torch.no_grad():
        for x, in dataloader:
            out = model.infer(x)
            emb = torch.cat((out.RGB_appearance, out.RGB_structure), dim=-1)
            rec = models['RGB_decoder'](emb)

            x = (x.squeeze()[:3].permute(1, 2, 0).cpu().numpy() + 1) / 2
            rec = (rec.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(x)
            ax[1].imshow(rec)

            break
