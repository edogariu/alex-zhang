import os
from typing import Tuple
import numpy as np
from PIL import Image
import tqdm
from scipy.interpolate import griddata
import cv2

# for multiprocessing
from multiprocessing import Pool, Value
import sys
from os import cpu_count
import itertools
import time

import torch
import torch.utils.data as D

import radiate
from utils import TOP_DIR_NAME, SPLIT_INTERVALS

def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args
    # sys.stdout = open(os.devnull, 'w') 
    # sys.stderr = open(os.devnull, 'w')

def lidar_transform(lidar: np.ndarray):
    lidar = np.max(lidar, axis=-1)
    lidar[lidar != 0] = 255 - lidar[lidar != 0]
    
    rows, columns = lidar.shape
    coordinates_grid = np.ones((2, rows, columns), dtype=np.int32)
    coordinates_grid[0] = coordinates_grid[0] * np.array([range(rows)]).T
    coordinates_grid[1] = coordinates_grid[1] * np.array([range(columns)])
    mask = lidar != 0
    highest = np.argmax(mask.sum(axis=1) > 0)
    non_zero_coords = np.hstack((coordinates_grid[0][mask].reshape(-1, 1),
                                coordinates_grid[1][mask].reshape(-1, 1)))
    
    values = lidar[lidar != 0]
    xi = coordinates_grid.reshape(2, -1).T
    out = griddata(non_zero_coords, values, xi, method='nearest').reshape(rows, columns)
    out[np.isnan(out)] = 0.
    out[:highest] = 0.
    return np.expand_dims(out, axis=-1)

def crop(rgbd: np.ndarray,
         resolution: Tuple[int]):
    mid = rgbd.shape[1] // 2
    out = rgbd[-resolution[0] * 2:, mid - resolution[1]: mid + resolution[1]]
    out = cv2.resize(out, resolution)
    return out
    

class RadiateDataset:
    def __init__(self,
                 resolution: Tuple[int],
                 filepath: str=os.path.join(TOP_DIR_NAME, 'data', 'radiate'),
                 n_data = 10000,
                 use_multiprocessing: bool=True):

        assert os.path.isdir(os.path.join(filepath, 'config'))
        self.label_to_int = {'city': 0, 'fog': 1, 'night': 2, 'junction': 3, 'motorway': 4, 'rain': 5, 'snow': 6}
        self.int_to_label = ['city', 'fog', 'night', 'junction', 'motorway', 'rain', 'snow']
        
        print('DATASET INFO: Loading the RADIATE dataset at resolution {}'.format(resolution))
        
        if not os.path.isfile(os.path.join(filepath, 'rgbd_images_{}_{}.npy'.format(resolution, n_data))):
            rgbd_images = []
            scene_labels = []
            
            dirs = [dir for dir in np.sort(os.listdir(filepath)) if dir.split('_')[0] in self.label_to_int]
            
            if use_multiprocessing:
                # split da pie
                counter = Value('i', 0)
                n_cpu = cpu_count()
                n_jobs = min(n_cpu, len(dirs))
                ns = [n_data // (n_jobs - 1) for _ in range(n_jobs - 1)]
                ns.append(n_data - sum(ns))
                print('DATASET INFO: spawning {} processes to load this big ass dataset'.format(n_jobs))
                assert sum(ns) == n_data
                assert len(ns) == len(dirs)
                
                with Pool(processes=n_cpu, initializer = init, initargs = (counter, )) as pool:
                    ret = pool.starmap_async(self._load_dir, [(filepath, dir, n, resolution, use_multiprocessing) for dir, n in zip(dirs, ns)])

                    prev_count = counter.value
                    last_update = time.perf_counter()
                    with tqdm.tqdm(total=n_data) as pbar:
                        while counter.value < n_data:
                            if counter.value != prev_count:
                                prev_count = counter.value
                                pbar.update(1)
                                last_update = time.perf_counter()
                            else:
                                time.sleep(0.001)
                                if time.perf_counter() - last_update > 5: 
                                    print('DATASET INFO: for some reason we didnt get em all :(')
                                    break
                    pool.close()
                    pool.join()
                    ret = ret.get()
                    
                    rgbd_images = list(itertools.chain.from_iterable([r[0] for r in ret]))
                    scene_labels = list(itertools.chain.from_iterable([r[1] for r in ret]))
                    assert len(rgbd_images) > 0.9 * n_data, 'we missed more than 10%'
            else:
                rgbd_images = []
                scene_labels = []
                for dir in tqdm.tqdm(dirs):
                    r, s = self._load_dir(filepath, dir, n_data // len(dirs), use_multiprocessing)
                    rgbd_images.extend(r)
                    scene_labels.extend(s)
            
            rgbd_images = np.stack(rgbd_images)
            scene_labels = np.array(scene_labels)
            
            # normalize the images -- RGB between -1 and 1, and depth between -1 and 1 (ish)
            rgbd_images = rgbd_images.astype(float)
            rgbd_images[:, :, :, :3] /= 128.
            rgbd_images[:, :, :, :3] -= 1.
            rgbd_images[:, :, :, 3] /= 128.
            rgbd_images[:, :, :, 3] -= 1.
        
            # save the dataset for future loading
            np.save(os.path.join(filepath, 'rgbd_images_{}_{}.npy'.format(resolution, n_data)), rgbd_images)
            np.save(os.path.join(filepath, 'scene_labels_{}_{}.npy').format(resolution, n_data), scene_labels)
            print('DATASET INFO: saved dataset of {} datapoints as .npy files in {}'.format(len(rgbd_images), filepath))
            
        else:  # load the dataset
            rgbd_images = np.load(os.path.join(filepath, 'rgbd_images_{}_{}.npy'.format(resolution, n_data)))
            scene_labels = np.load(os.path.join(filepath, 'scene_labels{}_{}.npy').format(resolution, n_data))
            scene_labels = np.array([self.label_to_int[s] for s in scene_labels])
            
        # # inspect whats goin on
        # import matplotlib.pyplot as plt; 
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow((rgbd_images[0, :, :, :3] + 1) / 2); 
        # ax[1].imshow((rgbd_images[0, :, :, 3] + 1) / 2); 
        # plt.waitforbuttonpress(0); exit(0)
        # import matplotlib.pyplot as plt; plt.hist(rgbd_images[:, :, :, 3].reshape(-1)); plt.waitforbuttonpress(0); exit(0)  # inspect the depths
        
        # store the images as a tensor, to allow for the creation of dataloaders and such
        self.rgbd_images = torch.from_numpy(rgbd_images).float().permute(0, 3, 1, 2)  # NHWC -> NCHW
        self.scene_labels = torch.from_numpy(scene_labels.astype(int)).long()

        print('DATASET INFO: Loaded {} datapoints!'.format(len(self)))
        
    def _load_dir(self, filepath: str, dir: str, n: int, resolution: Tuple[int], using_multiprocessing: bool):
        if using_multiprocessing: global counter
        
        rgbd_images = []
        scene_labels = []
        seq = radiate.Sequence(os.path.join(filepath, dir), 
                                   config_file=os.path.join(filepath, 'config', 'config.yaml'),
                                   calib_file=os.path.join(filepath, 'config', 'default-calib.yaml'))

        pbar = np.linspace(seq.init_timestamp, seq.end_timestamp, n + 2)[1:-1] if using_multiprocessing else tqdm.tqdm(np.linspace(seq.init_timestamp, seq.end_timestamp, n + 2)[1:-1])
        for t in pbar:
            output = seq.get_from_timestamp(t)
            if 'sensors' not in output: continue
            output = output['sensors']
            if 'camera_right_rect' in output.keys() and 'proj_lidar_right' in output.keys():
                rgb = output['camera_right_rect']
                proj_lidar = output['proj_lidar_right']
                proj_lidar = lidar_transform(proj_lidar)
                rgbd = np.concatenate([rgb, proj_lidar], axis=-1)
                rgbd = crop(rgbd, resolution)
                rgbd_images.append(rgbd)
                
            if 'camera_left_rect' in output.keys() and 'proj_lidar_left' in output.keys():
                rgb = output['camera_left_rect']
                proj_lidar = output['proj_lidar_left']
                proj_lidar = lidar_transform(proj_lidar)
                rgbd = np.concatenate([rgb, proj_lidar], axis=-1)
                rgbd = crop(rgbd, resolution)
                rgbd_images.append(rgbd)
                scene_labels.append(dir.split('_')[0])
            
            if using_multiprocessing:
                with counter.get_lock():
                    counter.value += 1
                    
        return rgbd_images, scene_labels
    
    def __len__(self) -> int:
        return len(self.rgbd_images)
    
    def get_dataloader(self, 
                       split: str, 
                       batch_size: int,
                       shuffle: bool=True,
                       drop_last: bool=True,
                       include_scene_labels: bool=False):
        
        assert split in ['train', 'val', 'test', 'all']
        
        # split the dataset the same way every time
        np.random.seed(0)
        rand_idxs = np.random.permutation(len(self)) if shuffle else np.arange(len(self))
        val_idx = int(len(rand_idxs) * SPLIT_INTERVALS['train']); test_idx = val_idx + int(len(rand_idxs) * SPLIT_INTERVALS['val'])
        train_idxs, val_idxs, test_idxs = rand_idxs[:val_idx], rand_idxs[val_idx: test_idx], rand_idxs[test_idx:]
        idxs = {'train': train_idxs,
                     'val': val_idxs,
                     'test': test_idxs,
                     'all': rand_idxs}[split]
        np.random.seed()
        
        rgbd_images = self.rgbd_images[idxs]
        if not include_scene_labels: 
            return D.DataLoader(D.TensorDataset(rgbd_images), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        else:
            scene_labels = self.scene_labels[idxs]
            return D.DataLoader(D.TensorDataset(rgbd_images, scene_labels), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

class RGBDDataset:
    def __init__(self,
                 resolution: Tuple[int],
                 filepath: str=os.path.join(TOP_DIR_NAME, 'data', 'rgbd-scenes-v2')):

        print('DATASET INFO: Loading the RGB-D objects dataset at resolution {}'.format(resolution))

        if not os.path.isfile(os.path.join(filepath, 'rgbd_images_{}.npy'.format(resolution))):
            # load images and labels from the dataset
            rgbd_images = []
            scene_labels = []
            for scene in tqdm.tqdm(np.sort(os.listdir(os.path.join(filepath, 'imgs')))):
                if not 'scene' in scene: continue
                scene_label = scene.split('_')[1]
                scene_dir_path = os.path.join(filepath, 'imgs', scene)
                
                idxs = np.sort(np.unique([f.split('-')[0] for f in os.listdir(scene_dir_path)]))  # get unique indices for each image
                for idx in idxs:
                    color_path = os.path.join(scene_dir_path, '{}-color.png'.format(idx))
                    depth_path = os.path.join(scene_dir_path, '{}-depth.png'.format(idx))
                    if not os.path.isfile(color_path) or not os.path.isfile(depth_path): continue
                    color = np.array(Image.open(color_path).resize(resolution))  # load color image
                    depth = np.expand_dims(np.array(Image.open(depth_path).resize(resolution)), axis=-1)  # load depth image
                    img = np.concatenate((color, depth), axis=-1)  # concat the images
                    rgbd_images.append(img)
                    scene_labels.append(scene_label)
            
            rgbd_images = np.stack(rgbd_images)
            scene_labels = np.array(scene_labels)
            
            # normalize the images -- RGB between -1 and 1, and depth between -1 and 1 (ish)
            rgbd_images = rgbd_images.astype(float)
            rgbd_images[:, :, :, :3] /= 128.
            rgbd_images[:, :, :, :3] -= 1.
            rgbd_images[:, :, :, 3] /= 10000
            rgbd_images[:, :, :, 3] -= 1
            
            # save the dataset for future loading
            np.save(os.path.join(filepath, 'rgbd_images_{}.npy'.format(resolution)), rgbd_images)
            np.save(os.path.join(filepath, 'scene_labels_{}.npy').format(resolution), scene_labels)
            print('DATASET INFO: saved dataset of {} datapoints as .npy files in {}'.format(len(rgbd_images), filepath))

        else:
            # load the dataset
            rgbd_images = np.load(os.path.join(filepath, 'rgbd_images_{}.npy'.format(resolution)))
            scene_labels = np.load(os.path.join(filepath, 'scene_labels_{}.npy').format(resolution))
            # import matplotlib.pyplot as plt; plt.hist(rgbd_images[:, :, :, 3].reshape(-1)); plt.waitforbuttonpress(0); exit(0)  # inspect the depths
        
        # store the images as a tensor, to allow for the creation of dataloaders and such
        self.rgbd_images = torch.from_numpy(rgbd_images).float().permute(0, 3, 1, 2)  # NHWC -> NCHW
        self.scene_labels = torch.from_numpy(scene_labels.astype(int)).long()

        print('DATASET INFO: Loaded {} datapoints!'.format(len(self)))
        
    def __len__(self) -> int:
        return len(self.rgbd_images)
    
    def get_dataloader(self, 
                       split: str, 
                       batch_size: int,
                       shuffle: bool=True,
                       drop_last: bool=True,
                       include_scene_labels: bool=False):
        
        assert split in ['train', 'val', 'test', 'all']
        
        # split the dataset the same way every time
        np.random.seed(0)
        rand_idxs = np.random.permutation(len(self)) if shuffle else np.arange(len(self))
        val_idx = int(len(rand_idxs) * SPLIT_INTERVALS['train']); test_idx = val_idx + int(len(rand_idxs) * SPLIT_INTERVALS['val'])
        train_idxs, val_idxs, test_idxs = rand_idxs[:val_idx], rand_idxs[val_idx: test_idx], rand_idxs[test_idx:]
        idxs = {'train': train_idxs,
                     'val': val_idxs,
                     'test': test_idxs,
                     'all': rand_idxs}[split]
        np.random.seed()
        
        rgbd_images = self.rgbd_images[idxs]
        if not include_scene_labels: 
            return D.DataLoader(D.TensorDataset(rgbd_images), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        else:
            scene_labels = self.scene_labels[idxs]
            return D.DataLoader(D.TensorDataset(rgbd_images, scene_labels), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

if __name__ == '__main__':
    resolution = (128, 128)
    d = RadiateDataset(resolution, n_data=100, use_multiprocessing=True)
    print(len(d))
    dl = d.get_dataloader('all', 80)
    for x, in dl:
        print(x.shape)
        break
    