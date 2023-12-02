# LoRA Module
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from torch.optim import Adam
from dataset import collate_fn
from tqdm import tqdm

def calculateIoU(pred, gt):
    intersect = (pred * gt).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + gt.sum(dim=(-1, -2)) - intersect
    ious = intersect.div(union)
    return ious

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)

    @property
    def weight(self):
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return 0

class MonkeyPatchLoRALinear(nn.Module):
    # It's "monkey patch" means you can replace nn.Linear with the new
    # LoRA Linear class without modifying any other code.
    def __init__(self, fc: nn.Linear, rank=4, lora_scale=1):
        super().__init__()
        if rank > min(fc.in_features, fc.out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(fc.in_features, fc.out_features)}"
            )
        if not isinstance(fc, nn.Linear):
            raise ValueError(
                f"MonkeyPatchLoRALinear only support nn.Linear, but got {type(fc)}"
            )

        self.fc = fc
        self.rank = rank
        self.lora_scale = lora_scale

        in_features = fc.in_features
        out_features = fc.out_features
        self.fc_lora = LoRALinearLayer(in_features, out_features, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc(hidden_states) + \
                        self.lora_scale * self.fc_lora(hidden_states)
        return hidden_states

    @property
    def weight(self):
        return self.fc.weight + self.lora_scale * self.fc_lora.weight

    @property
    def bias(self):
        return self.fc.bias

# your implementation

class LoRAConv2DLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride = 1, padding = 0, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )
        
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel
        self.rank = rank

        self.down = nn.Conv2d(in_features, rank, 1, 1, 0, bias=False)
        self.up = nn.Conv2d(rank, out_features, kernel, stride, padding, bias=False)

        nn.init.normal_(self.down.weight, std = 1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)
    
    @property
    def weight(self):
        composite_weight = torch.einsum('rnjk,mrjk->mnjk', self.up.weight, self.down.weight)
        return composite_weight

    @property
    def bias(self):
        return 0

class MonkeyPatchLoRAConv2D(nn.Module):
    def __init__(self, module: nn.Conv2d, rank=4, lora_scale=1):
        super().__init__()
        if rank > min(module.in_channels, module.out_channels):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(module.in_channels, module.out_channels)}"
            )
        if not isinstance(module, nn.Conv2d):
            raise ValueError(
                f"MonkeyPatchLoRALinear only support nn.Linear, but got {type(module)}"
            )

        self.conv = module
        self.rank = rank
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
        self.lora_scale = lora_scale

        in_channels = module.in_channels
        out_channels = module.out_channels
        self.conv_lora = LoRAConv2DLayer(in_channels, out_channels, self.kernel_size, self.stride, self.padding, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states) + \
                        self.lora_scale * self.conv_lora(hidden_states)
        return hidden_states
    
    @property
    def weight(self):
        return self.conv.weight + self.lora_scale * self.conv_lora.weight

    @property
    def bias(self):
        return self.conv.bias

# class LoRAConvTranspose2DLayer(nn.Module):
#     ...

# class MonkeyPatchLoRAConvTranspose2D(nn.Module):
#     ...

def replace_LoRA(model:nn.Module, cls):
    for name, block in model.named_children():
        # patch every nn.Linear in Mlp
        if isinstance(block, nn.Linear) and cls == MonkeyPatchLoRALinear:
            block = cls(block, 4, 1)
            setattr(model, name, block)

            for param in block.fc.parameters():
                param.requires_grad_(False)
            for param in block.fc_lora.parameters():
                param.requires_grad_(True)
        
        elif isinstance(block, nn.Conv2d) and cls == MonkeyPatchLoRAConv2D:
            min_channel = min(block.in_channels, block.out_channels)
            if min_channel > 4:
                block = cls(block, 4, 1)
                setattr(model, name, block)

                for param in block.conv.parameters():
                    param.requires_grad_(False)
                for param in block.conv_lora.parameters():
                    param.requires_grad_(True)
                    
        else:
            replace_LoRA(block, cls)

def print_params(model):
    model_parameters = filter(lambda p: True, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("total params: ", params)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("training params: ", params)

def downsamle_SAM(sam, opt):
    """ Downsample SAM by taking the first 1/4 pe  """
    
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

    # inject LoRA
    LoRA_sam = deepcopy(downsampled_sam)
    for param in LoRA_sam.parameters():
        param.requires_grad_(False)

    if opt.linear:
        replace_LoRA(LoRA_sam.mask_decoder, MonkeyPatchLoRALinear)

    if opt.conv2d:
        replace_LoRA(LoRA_sam, MonkeyPatchLoRAConv2D)

    print_params(LoRA_sam)
    return LoRA_sam

def lowres_SAM(sam, opt, train_dataset, val_dataset):
    sam.cuda()
    
    patchify_model = nn.Conv2d(3, 768, kernel_size=(4, 4), stride=(4, 4))
    patchify_model.cuda()
    upsampler = torch.nn.Upsample((1024, 1024))

    dist_lr = 1e-3
    optimizer = Adam(patchify_model.parameters(), dist_lr)
    finetune_epoch = 10

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, collate_fn=collate_fn, 
                                            num_workers=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn, 
                                            num_workers=8, shuffle=False)

    print(" Dist Start ... ")
    for epoch in range(finetune_epoch): 
        # training 
        patchify_model.train()
        avg_loss = []
        for batch_idx, (batch_data, target) in tqdm(enumerate(train_loader)):
            
            images = torch.stack([k["image"] for k in batch_data], dim=0).cuda()
            padh = 256 - images.shape[2]
            padw = 256 - images.shape[3]
            images = F.pad(images, (0, padw, 0, padh))

            # Upsample images
            upsampled = upsampler(images)
            target = sam.image_encoder.patch_embed.proj(upsampled)

            pred = patchify_model(images)
            loss = torch.mean( (pred - target) ** 2 )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss.append(loss.item())

            if batch_idx % 50 == 0:
                print(" Dist Epoch {} / {};  Avg Loss {} ".format(batch_idx, epoch, torch.tensor(avg_loss).mean()))

        avg_loss = torch.tensor(avg_loss).mean()
        print(" Dist Epoch {} avg loss {}".format(epoch, avg_loss.item()))
    
    print(" Testing ... ")
    sam.image_encoder.patch_embed.proj = patchify_model
    sam.image_encoder.img_size = 256

    with torch.no_grad():
        total_ious = torch.tensor([], device="cuda:0")
        for batch_idx, (batch_data, target) in tqdm(enumerate(test_loader)):
            target = [t.cuda() for t in target]
        
            update_batches = []
            for idx, batch in enumerate(batch_data):
                update_batch = {k:v.cuda() for k, v in batch.items()}
                update_batch["original_size"] = (160, 256)
                update_batches.append(update_batch)

            pred = sam(update_batches, multimask_output=False)
            pred_mask = [p["masks"].squeeze(1) for p in pred] 
            
            for p, t in zip(pred_mask, target):
                ious = calculateIoU(p, t)
                total_ious = torch.cat([total_ious, ious])
        
        mean_ious = total_ious.mean()
        print("TEST total masks {} mIoU {}".format(len(total_ious), mean_ious.item()))

    import pdb; pdb.set_trace()

    LoRA_sam = deepcopy(sam)
    for param in LoRA_sam.parameters():
        param.requires_grad_(False)

    if opt.linear:
        replace_LoRA(LoRA_sam.mask_decoder, MonkeyPatchLoRALinear)

    if opt.conv2d:
        replace_LoRA(LoRA_sam, MonkeyPatchLoRAConv2D)

    print_params(LoRA_sam)
    return LoRA_sam

