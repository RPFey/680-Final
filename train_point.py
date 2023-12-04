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
from dataset import SA1B_Dataset, SA1bSubset, input_transforms, target_transforms, SA1bSubsetPoint
from dataset import show_points, show_mask, collate_fn, collate_fn_point
from LoRA import downsamle_SAM, calculateIoU, lowres_SAM
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

def test(model: Sam, test_dataset: SA1bSubset, opt):
    model.cuda()
    ckpts = glob.glob(os.path.join(opt.save_dir, opt.expname, "*.pt"))
    # Load the latest checkpoint for testing
    if len(ckpts) > 0:
        ckpts.sort()
        latest_ckpt = ckpts[-1]
        states = torch.load(latest_ckpt, map_location=torch.device("cuda:0"))
        model.load_state_dict(states["model_state"])
    else:
        raise Exception(" No ckpts found in {} ".format(os.path.join(opt.save_dir, opt.expname)))
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn_point, 
                                            num_workers=8, shuffle=False)
    
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
        
        mean_ious = total_ious.mean()
        print("TEST total masks {} mIoU {}".format(len(total_ious), mean_ious.item()))

def train(model: Sam, train_dataset: SA1bSubset, test_dataset: SA1bSubset, opt, summary_writer: SummaryWriter):
    model.cuda()
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=opt.lr)

    # check existing data
    writer_dir = os.path.join(opt.save_dir, opt.expname)
    ckpts = glob.glob(os.path.join(writer_dir, "*.pt"))
    if len(ckpts) > 0:
        ckpts.sort()
        latest_ckpt = ckpts[-1]
        states = torch.load(latest_ckpt, map_location=torch.device("cuda:0"))
        model.load_state_dict(states["model_state"])
        optimizer.load_state_dict(states["optimizer"])
        start_epoch = states["epoch"]
        global_step = states["global_step"]
    else:
        start_epoch = 0
        global_step = 0
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, collate_fn=collate_fn_point, 
                                            num_workers=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn_point, 
                                            num_workers=8, shuffle=False)
    for epoch in range(start_epoch, opt.num_epochs): 
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
                    points_cpu = update_batches[0]["point_coords"].cpu()
                    point_labels_cpu = update_batches[0]["point_labels"].cpu()

                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    for axis, mask in zip(axes, masks):
                        axis.imshow(image_cpu / 255.)
                        for m in mask[4:7]:
                            show_mask(m, axis, random_color=True)
                        for p, p_l in zip(points_cpu, point_labels_cpu):
                            show_points(p, p_l, axis)
                    
                    summary_writer.add_figure("test_{}".format(batch_idx), fig, epoch)
            
            mean_ious = total_ious.mean()
            print("TEST EPOCH {}, mIoU {}".format(epoch, mean_ious.item()))
            summary_writer.add_scalar("test/mIoU", mean_ious.item(), epoch)

        # training 
        model.train()
        for batch_idx, (batch_data, target) in tqdm(enumerate(train_loader)):
            
            target = torch.stack(target, dim=0).cuda()
            images = torch.stack([k["image"] for k in batch_data], dim=0).cuda()
            point_coords = torch.stack([k["point_coords"] for k in batch_data], dim=0).cuda()

            pred = model.batch_forward_points(images, point_coords, (160, 256), multimask_output=False)

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
                    axis.imshow(image_cpu / 255.)
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
            name = os.path.join(writer_dir, "LoRA_sam_{}.pt".format(epoch))
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step
            }

            torch.save(ckpt, name)
    
    # Save Model at the end
    name = os.path.join(writer_dir, "LoRA_sam_{}.pt".format(epoch))
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step
    }
    torch.save(ckpt, name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./logs")
    parser.add_argument("--expname", type=str, required=True)
    parser.add_argument("--linear", action="store_true", default=False)
    parser.add_argument("--conv2d", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--method", type=str, default="downsample")
    parser.add_argument("--data_dir", type=str, default="./sa1b")

    opt = parser.parse_args()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    writer_dir = os.path.join(opt.save_dir, opt.expname)
    summary_writer = SummaryWriter(writer_dir)
    
    path = './sa1b'
    dataset = SA1B_Dataset(root=path, transform=input_transforms, target_transform=target_transforms)
    all_index = np.arange(len(dataset))
    # np.random.shuffle(all_index)
    train_num = int(0.8 * len(dataset))

    # 128 as val index
    train_index = all_index[:train_num-128]
    val_index = all_index[train_num-128:train_num]
    test_index = all_index[train_num:]

    print("Loading Datasets ...")
    train_dataset = SA1bSubsetPoint(train_index, num_points=2, is_test=False, root=path, 
                                transform=input_transforms, target_transform=target_transforms)
    val_dataset = SA1bSubsetPoint(val_index, num_points=2, is_test=False, root=path, 
                                transform=input_transforms, target_transform=target_transforms)
    test_dataset = SA1bSubsetPoint(test_index, num_points=2, is_test=True, root=path, 
                                transform=input_transforms, target_transform=target_transforms)
    
    # Copy SAM Model
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    if opt.method == "downsample":
        LoRA_sam = downsamle_SAM(sam, opt)
    elif opt.method == "lowres":
        LoRA_sam = lowres_SAM(sam, opt, train_dataset, val_dataset)
    else:
        raise NotImplementedError(" Unimplemented {} ".format(opt.method))

    train(LoRA_sam, train_dataset, val_dataset, opt, summary_writer)
    test(LoRA_sam, test_dataset, opt)
