import os
import glob
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

from PIL import Image
import matplotlib.pyplot as plt

from pycocotools import mask as mask_utils

import json
import numpy as np
from tqdm import tqdm

# SAM on Low resolution images
from segment_anything import SamPredictor
from segment_anything.modeling.sam import Sam
from segment_anything import sam_model_registry
from dataset import SA1B_Dataset, SA1bSubset, input_transforms, target_transforms
from dataset import show_box, show_mask
from LoRA import MonkeyPatchLoRAConv2D, MonkeyPatchLoRALinear, replace_LoRA, downsamle_SAM
from copy import deepcopy
import argparse

from torch.optim import Adam
import random
import pdb
from torch.utils.tensorboard import SummaryWriter

def compute_loss(pred, gt_mask, alpha = 0.25, gamma = 2):
    h, w = gt_mask.shape[-2:]
    gt_mask = gt_mask.reshape(-1, h, w)

    pred_mask = torch.sigmoid(pred[:, -1, :, :])
    fl_1 = -alpha * ( (1 - pred_mask[gt_mask > .5]) ** gamma ) * \
        torch.log(pred_mask[gt_mask > .5] + 1e-6)
    
    fl_2 = -(1-alpha) * ( (pred_mask[gt_mask < .5]) ** gamma ) * \
        torch.log(1 - pred_mask[gt_mask < .5] + 1e-6)
    
    focal_loss = (torch.mean(fl_1) + torch.mean(fl_2))
    
    dice_loss = 2 * torch.sum( pred_mask * gt_mask, dim=(-1, -2) ) / \
        ( torch.sum( pred_mask ** 2, dim=(-1, -2) ) + torch.sum( gt_mask ** 2, dim=(-1, -2) ) + 1e-5)
    dice_loss = (1 - dice_loss)
    dice_loss = torch.mean(dice_loss)
    
    return focal_loss, dice_loss

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./logs")
parser.add_argument("--expname", type=str, required=True)
parser.add_argument("--linear", action="store_true", default=False)
parser.add_argument("--conv2d", action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--data_dir", type=str, default="./sa1b")

opt = parser.parse_args()

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

writer_dir = os.path.join(opt.save_dir, opt.expname)
summary_writer = SummaryWriter(writer_dir)

# Main Training Loop
num_epochs = 10
batch_size = opt.batch_size
lr = 1e-3

def collate_fn(batches):
    batch_data = []
    targets = []

    for b in batches:
        image, target, bbox = b
        batch_data.append(
            {
                "image": image,
                "boxes": bbox
            }
        )

        targets.append(target)

    return batch_data, targets

def calculateIoU(pred, gt):
    intersect = (pred * gt).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + gt.sum(dim=(-1, -2)) - intersect
    ious = intersect.div(union)
    return ious

def test(model: Sam, train_dataset: SA1bSubset, test_dataset: SA1bSubset):
    model.cuda()
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn, 
                                            num_workers=8, shuffle=False)
    global_step = 0

    model.eval()
    with torch.no_grad():
        total_ious = torch.tensor([], device="cuda:0")
        for batch_idx, (batch_data, target) in tqdm(enumerate(test_loader)):
            target = [t.cuda() for t in target]
        
            update_batches = []
            for idx, batch in enumerate(batch_data):
                update_batch = {k:v.cuda() for k, v in batch.items()}
                update_batch["original_size"] = (160, 256)
                update_batches.append(update_batch)

            pred = model(update_batches, multimask_output=False)
            pred_mask = [p["masks"].squeeze(1) for p in pred] 
            
            for p, t in zip(pred_mask, target):
                ious = calculateIoU(p, t)
                total_ious = torch.cat([total_ious, ious])

            # show results
            if batch_idx in [0, 5, 10]:
                masks = [pred_mask[0].cpu().numpy(), target[0].cpu().numpy()]
                image_cpu = update_batches[0]["image"].cpu().permute(1, 2, 0)
                box_cpu = update_batches[0]["boxes"].cpu()

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                for axis, mask in zip(axes, masks):
                    axis.imshow(image_cpu)
                    for m in mask:
                        show_mask(m, axis, random_color=True)
                    for b in box_cpu:
                        show_box(b, axis)
                
                summary_writer.add_figure("test_{}".format(batch_idx), fig, epoch)
        
        mean_ious = total_ious.mean()
        print("TEST EPOCH {}, mIoU {}".format(epoch, mean_ious.item()))
        summary_writer.add_scalar("test/mIoU", mean_ious.item(), epoch)

if __name__ == "__main__":

    # Copy SAM Model
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    LoRA_sam = downsamle_SAM(sam, opt)

    path = './sa1b'
    dataset = SA1B_Dataset(root=path, transform=input_transforms, target_transform=target_transforms)
    all_index = np.arange(len(dataset))
    np.random.shuffle(all_index)
    train_num = int(0.8 * len(dataset))
    train_index = all_index[:-128]
    test_index = all_index[-128:]

    print("Loading Datasets ...")
    train_dataset = SA1bSubset(train_index, is_train=True, root=path, 
                                transform=input_transforms, target_transform=target_transforms)
    test_dataset = SA1bSubset(test_index, is_train=False, root=path, 
                                transform=input_transforms, target_transform=target_transforms)

    train(LoRA_sam, train_dataset, test_dataset)
