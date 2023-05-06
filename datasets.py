import os
from typing import Tuple
import numpy as np
from PIL import Image
import tqdm

import torch
import torch.utils.data as D

from utils import TOP_DIR_NAME, SPLIT_INTERVALS


class RGBDDataset:
    def __init__(self,
                 resolution: Tuple[int],
                 filepath: str=os.path.join(TOP_DIR_NAME, 'data', 'rgbd-scenes-v2')):

        print('DATASET INFO: Loading the RGB-D objects dataset at resolution {}'.format(resolution))

        # # load images and labels from the dataset
        # rgbd_images = []
        # scene_labels = []
        # for scene in tqdm.tqdm(np.sort(os.listdir(os.path.join(filepath, 'imgs')))):
        #     if not 'scene' in scene: continue
        #     scene_label = scene.split('_')[1]
        #     scene_dir_path = os.path.join(filepath, 'imgs', scene)
            
        #     idxs = np.sort(np.unique([f.split('-')[0] for f in os.listdir(scene_dir_path)]))  # get unique indices for each image
        #     for idx in idxs:
        #         color_path = os.path.join(scene_dir_path, '{}-color.png'.format(idx))
        #         depth_path = os.path.join(scene_dir_path, '{}-depth.png'.format(idx))
        #         if not os.path.isfile(color_path) or not os.path.isfile(depth_path): continue
        #         color = np.array(Image.open(color_path).resize(resolution))  # load color image
        #         depth = np.expand_dims(np.array(Image.open(depth_path).resize(resolution)), axis=-1)  # load depth image
        #         img = np.concatenate((color, depth), axis=-1)  # concat the images
        #         rgbd_images.append(img)
        #         scene_labels.append(scene_label)
        
        # rgbd_images = np.stack(rgbd_images)
        # scene_labels = np.array(scene_labels)
        
        # # normalize the images -- RGB between -1 and 1, and depth between -1 and 1 (ish)
        # rgbd_images = rgbd_images.astype(float)
        # rgbd_images[:, :, :, :3] /= 128.
        # rgbd_images[:, :, :, :3] -= 1.
        # rgbd_images[:, :, :, 3] /= 10000
        # rgbd_images[:, :, :, 3] -= 1
        
        # # save the dataset for future loading
        # np.save(os.path.join(filepath, 'rgbd_images_{}.npy'.format(resolution)), rgbd_images)
        # np.save(os.path.join(filepath, 'scene_labels.npy'), scene_labels)
        # print('DATASET INFO: saved dataset of {} datapoints as .npy files in {}'.format(len(rgbd_images), filepath))
        # exit(0)
        
        # load the dataset
        rgbd_images = np.load(os.path.join(filepath, 'rgbd_images_{}.npy'.format(resolution)))
        scene_labels = np.load(os.path.join(filepath, 'scene_labels.npy'))
        # import matplotlib.pyplot as plt; plt.hist(rgbd_images[:, :, :, 3].reshape(-1)); plt.waitforbuttonpress(0); exit(0)  # inspect the depths
        
        # store the images as a tensor, to allow for the creation of dataloaders and such
        self.rgbd_images = torch.from_numpy(rgbd_images).float().permute(0, 3, 1, 2)  # NHWC -> NCHW
        self.scene_labels = scene_labels

        print('DATASET INFO: Loaded {} datapoints!'.format(len(self.rgbd_images)))
        
    def __len__(self) -> int:
        return len(self.rgbd_images)
    
    def get_dataloader(self, 
                       split: str, 
                       batch_size: int,
                       shuffle: bool=True,
                       drop_last: bool=True):
        
        assert split in ['train', 'val', 'test', 'all']
        
        # split the dataset the same way every time
        np.random.seed(0)
        rand_idxs = np.random.permutation(len(self))
        val_idx = int(len(rand_idxs) * SPLIT_INTERVALS['train']); test_idx = val_idx + int(len(rand_idxs) * SPLIT_INTERVALS['val'])
        train_idxs, val_idxs, test_idxs = rand_idxs[:val_idx], rand_idxs[val_idx: test_idx], rand_idxs[test_idx:]
        idxs = {'train': train_idxs,
                     'val': val_idxs,
                     'test': test_idxs,
                     'all': rand_idxs}[split]
        np.random.seed()
        
        rgbd_images = self.rgbd_images[idxs]
        # scene_labels = self.scene_labels[idxs]
        
        return D.DataLoader(D.TensorDataset(rgbd_images), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

if __name__ == '__main__':
    resolution = (64, 64)
    d = RGBDDataset(resolution)
    dl = d.get_dataloader('all', 256)
    for x, in dl:
        print(x.shape)
        break
    