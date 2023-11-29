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
from LoRA import MonkeyPatchLoRAConv2D, MonkeyPatchLoRALinear, replace_LoRA
from copy import deepcopy
import argparse

def create_sampled_grid(scales, orig_shape):
    y_scale, x_scale = scales
    y_shape, x_shape = orig_shape

    sampled_coord = torch.meshgrid(torch.arange(0, y_shape, y_scale).float(), 
            torch.arange(0, x_shape, x_scale).float())
    
    sampled_coord = torch.stack(sampled_coord, dim=-1)
    print(sampled_coord[..., 0], sampled_coord[..., 1])
    sampled_coord[..., 0] = (sampled_coord[..., 0] - y_shape / 2) / y_shape / 2
    sampled_coord[..., 1] = (sampled_coord[..., 1] - x_shape / 2) / x_shape / 2

    return sampled_coord

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

def train(model: Sam, train_dataset: SA1bSubset, test_dataset: SA1bSubset):
    model.cuda()
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    # check existing data
    ckpts = glob.glob(os.path.join(opt.save_dir, opt.expname, "*.pt"))
    if len(ckpts) > 0:
        ckpts.sort()
        latest_ckpt = ckpts[-1]
        states = torch.load(os.path.join(opt.save_dir, opt.expname, latest_ckpt), map_location=torch.device("cuda:0"))
        model.load_state_dict(states["model_state"])
        optimizer.load_state_dict(states["optimizer"])
        start_epoch = states["epoch"]
    else:
        start_epoch = 0
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, 
                                            num_workers=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn, 
                                            num_workers=8, shuffle=False)
    global_step = 0

    for epoch in range(start_epoch, num_epochs): 
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

        # training 
        model.train()
        for batch_idx, (batch_data, target) in tqdm(enumerate(train_loader)):
            
            target = torch.stack(target, dim=0).cuda()
            images = torch.stack([k["image"] for k in batch_data], dim=0).cuda()
            boxes = torch.stack([k["boxes"] for k in batch_data], dim=0).cuda()

            pred = model.batch_forward_box(images, boxes, (160, 256), multimask_output=False)

            focal_loss, dice_loss = compute_loss(pred, target)
            loss = focal_loss + 0.01 * dice_loss

            with torch.no_grad():
                binary_mask = pred > model.mask_threshold
                binary_mask = binary_mask[:, 0, :, :]
                h, w = target.shape[-2:]
                target_mask = target.reshape(-1, h, w)

                intersect = (binary_mask * target_mask).sum(dim=(-1, -2))
                union = binary_mask.sum(dim=(-1, -2)) + target_mask.sum(dim=(-1, -2)) - intersect
                ious = intersect.div(union)
                train_ious = torch.mean(ious)

            if batch_idx % 50 == 0:
                print("ITER [{}] / EPOCH {}, loss: {}, focal: {}, dice: {}".format(global_step, epoch, loss.item(), focal_loss.item(), dice_loss.item()))

                # visualize the first image
                masks = [binary_mask[:10].cpu().numpy(), target_mask[:10].cpu().numpy()]
                image_cpu = images[0].cpu().permute(1, 2, 0)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                for axis, mask in zip(axes, masks):
                    axis.imshow(image_cpu)
                    for m in mask:
                        show_mask(m, axis, random_color=True)

                summary_writer.add_figure("train_{}".format(batch_idx), fig, epoch)
            
            summary_writer.add_scalar("train/loss", loss.item(), global_step)
            summary_writer.add_scalar("train/focal", focal_loss.item(), global_step)
            summary_writer.add_scalar("train/dice", dice_loss.item(), global_step)
            summary_writer.add_scalar("train/mIoU", train_ious.item(), global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        if epoch % 2 == 0 and epoch > 1:
            name = os.path.join(writer_dir, "LoRA_sam_{:03d}.pt".format(epoch))
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(name, ckpt)
    
    # Save Model at the end
    name = os.path.join(writer_dir, "final.pt")
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(name, ckpt)

# Copy SAM Model
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
downsampled_sam = deepcopy(sam)

# Since SAM Resolution is 1024 x 1024 \ Current input resolution is 160 x 256
# Y axis - Downsample 6.4 \ X axis - Downsample 4
filter_shape = sam.image_encoder.patch_embed.proj.weight.data.shape[-2:]

# hack SAM PE module
_, H, W, _ = sam.image_encoder.pos_embed.shape
pos_embed_feature = sam.image_encoder.pos_embed.data[:, :H//4, :W//4, :]

del downsampled_sam.image_encoder.pos_embed
downsampled_sam.image_encoder.pos_embed = nn.Parameter(pos_embed_feature)

# Test on low resolution
downsampled_sam.image_encoder.img_size = 256
downsampled_sam.prompt_encoder.factor = 4
downsampled_sam.prompt_encoder.image_embedding_size = (16, 16)

# remove downsampled SAM
# del downsampled_sam

def print_params(model):
    model_parameters = filter(lambda p: True, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("total params: ", params)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("training params: ", params)

# inject LoRA
LoRA_sam = deepcopy(downsampled_sam)
for param in LoRA_sam.parameters():
    param.requires_grad_(False)

if opt.linear:
    replace_LoRA(LoRA_sam.mask_decoder, MonkeyPatchLoRALinear)

if opt.conv2d:
    replace_LoRA(LoRA_sam, MonkeyPatchLoRAConv2D)

print_params(LoRA_sam)

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
