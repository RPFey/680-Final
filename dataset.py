import glob
import os
from typing import Any, Callable, Optional

from torchvision.datasets.folder import default_loader
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

from PIL import Image

from pycocotools import mask as mask_utils

import json
import numpy as np
from tqdm import tqdm

NUM_MASK_PER_IMG = 16

input_transforms = transforms.Compose([
    transforms.Resize((160, 256), antialias=True),
    transforms.ToTensor(),
])

target_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 256), antialias=True),
])

class SA1B_Dataset(torchvision.datasets.ImageFolder):
    """A data loader for the SA-1B Dataset from "Segment Anything" (SAM)
    This class inherits from :class:`~torchvision.datasets.ImageFolder` so
    the same methods can be overridden to customize the dataset.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, is_train=True, **kwargs):
        super(SA1B_Dataset, self).__init__(**kwargs)
        self.is_train = is_train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.imgs[index] # discard automatic subfolder labels
        image = self.loader(path)
        
        # Extract Labels
        target = []
        annos = json.load(open(f'{path[:-3]}json'))
        masks = annos['annotations'] # load json masks
        num_masks = len(masks)
        
        # For training split, we randomly select NUM_MASK_PER_IMG masks
        if num_masks >= NUM_MASK_PER_IMG : # and self.is_train :
            all_mask_index = np.arange(num_masks)
            np.random.shuffle(all_mask_index)
            select_mask_indices = all_mask_index[:NUM_MASK_PER_IMG]
        else:
            select_mask_indices = np.arange(num_masks)

        # Select only 
        for ind in select_mask_indices:
            m = masks[ind]
            # decode masks from COCO RLE format
            target.append(mask_utils.decode(m['segmentation']))

        target = np.stack(target, axis=-1)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        target[target > 0] = 1 # convert to binary masks

        bbox = []
        for mask in target:
            mask_y, mask_x = torch.where(mask > 0)
            bbox.append(torch.tensor([mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()]))
        bbox = torch.stack(bbox, dim=0)

        return image, target, bbox

    def __len__(self):
        return len(self.imgs)
    
    def extract_zero_label(self):
        """ Search for samples without any labels """
        zero_index = []
        
        for index in tqdm(range(len(self))):
            path, _ = self.imgs[index]
            annos = json.load(open(f'{path[:-3]}json'))
            masks = annos['annotations'] # load json masks
            if len(masks) == 0:
                zero_index.append(index)
            
        return zero_index

class SA1bSubset(SA1B_Dataset):
    def __init__(self, subset_index:np.ndarray, **kwargs):
        super(SA1bSubset, self).__init__(**kwargs)
        self.subset_index = subset_index

        assert np.max(subset_index) < super(SA1bSubset, self).__len__()
    
    def __len__(self):
        return len(self.subset_index)
    
    def __getitem__(self, index):
        # get data in the subset 
        sub_index = self.subset_index[index]
        img, target, bbox = super(SA1bSubset, self).__getitem__(sub_index)

        # Pad to NUM_MASK_PER_IMAGE
        if self.is_train and len(bbox) < NUM_MASK_PER_IMG:
            last_bbox = bbox[[-1]].repeat(NUM_MASK_PER_IMG - len(bbox), 1)
            bbox = torch.cat([bbox, last_bbox], dim=0)
            last_target = target[[-1]].repeat(NUM_MASK_PER_IMG - len(target), 1, 1)
            target = torch.cat([target, last_target], dim=0)

        return img, target, bbox
    
# !pip install git+https://github.com/facebookresearch/segment-anything.git
# you may want to make a local copy instead of pip install,
# as you may need to modify their code
# !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    