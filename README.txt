# Box Embedding
# LoRA on Linear Layer
python train.py --expname LoRA_SAM --linear 

# LoRA on Linear & Transpose Conv
python train.py --expname LoRA_SAM --linear --conv2d

# Point Embedding
# LoRA on Linear Layer
python train_point.py --expname LoRA_SAM --linear 

# LoRA on Linear & Transpose Conv
python train_point.py --expname LoRA_SAM --linear --conv2d
