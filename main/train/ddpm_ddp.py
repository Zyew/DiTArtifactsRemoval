import torch, os, random
import numpy as np
import json
from argparse import ArgumentParser
from time import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from PairedInputDataset import PairedInputDataset
from datetime import datetime
from diffusers.optimization import get_cosine_schedule_with_warmup
from collections import OrderedDict
from copy import deepcopy
from monai.data import DataLoader
from generative.networks.nets import VQVAE
from monai.transforms import Compose, Lambdad, ScaleIntensityRanged, RandAffined
from inference_utils_mix import denoise_batch, plot_triplets
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from diffusers import UNet2DModel, DDIMScheduler


## Reproducibility Setup
parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--train_timesteps_number", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--patch_size", type=int, default=2)
parser.add_argument("--model_dir", type=str, default="./ddpm_checkpoints")
parser.add_argument("--log_dir", type=str, default="./TrainlogFiles")
args = parser.parse_args()

seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# DDP init & device
os.environ["OMP_NUM_THREADS"] = "4"  
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high') 
dist.init_process_group(backend="nccl", init_method="env://")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank) 
device = torch.device("cuda", local_rank)
if torch.cuda.is_available():
    print(f"[Rank {rank}] Using GPU {torch.cuda.current_device()} out of {torch.cuda.device_count()} available GPUs")

## Model paths
epoch = args.epoch
learning_rate = args.learning_rate
T = args.train_timesteps_number
bsz = args.batch_size
psz = args.patch_size

training_start = datetime.now().strftime("%m%d_%H%M")  

model_dir = f"/home/hpc/iwi5/iwi5220h/DiTArtifactsRemoval/ddpm_checkpoints/ep{epoch}_bsz{bsz}_{training_start}"
os.makedirs(model_dir, exist_ok=True)

log_path = f"/home/hpc/iwi5/iwi5220h/DiTArtifactsRemoval/ddpm_checkpoints/trainlog_ep{epoch}_{training_start}.txt"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

## Model config
unet_config = {
    "sample_size": 64,        # latent H=W
    "in_channels": 2,         # noisy latent + cond latent
    "out_channels": 1,        # predict noise
    "layers_per_block": 2,
    "block_out_channels": (128, 256, 256, 256),
    "down_block_types": (
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D"
    ),
    "up_block_types": (
        "AttnUpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"
    ),
    "num_class_embeds": 4,  
    "class_embed_type": "timestep", 
}
if rank == 0:
    with open(os.path.join(args.model_dir, "unet_config.json"), "w") as f:
        json.dump(unet_config, f, indent=4)

model = UNet2DModel(**unet_config)

## Helper funciotn
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

ema = deepcopy(model).to(device)  
requires_grad(ema, False)
model = DDP(model.to(device), device_ids=[local_rank])

# Load pretrained vqvae
with open("./vqvae_with_perceptual/vqvae_config.json", "r") as f:
    config = json.load(f)

vae = VQVAE(**config)
vae.load_state_dict(torch.load("./vqvae_with_perceptual/best_vqvae.pth", map_location=device))
vae = vae.to(device)
vae.requires_grad_(False)
vae.eval()

## Data setup
train_transforms = Compose(
    [
        Lambdad(keys=["gt", "cond"], func=lambda x: x[None, ...]),
        ScaleIntensityRanged(keys=["gt", "cond"], a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True),
        RandAffined(
            keys=["gt", "cond"],
            spatial_size=[512, 512],
            prob=0.5 
        ),
    ]   
    )

data_dir = '/home/hpc/iwi5/iwi5220h/diffct/data/train'      
num_vis_samples = 4 
full_dataset = PairedInputDataset(data_dir, train_transforms)                                                
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size],
                                                 generator=torch.Generator().manual_seed(seed)
                                                )

def select_val_vis_samples(dataset, labels_to_find=[1,2,3,4]):
  
    found = {lab: None for lab in labels_to_find}
    needed = set(labels_to_find)
    for i in range(len(dataset)):
        sample = dataset[i]  
        lab = int(sample["label"].item()) if isinstance(sample["label"], torch.Tensor) else int(sample["label"])
        if lab in needed and found[lab] is None:
            found[lab] = sample
            needed.remove(lab)
        if not needed:
            break
    selected = []
    missing = []
    for lab in labels_to_find:
        if found[lab] is None:
            missing.append(lab)
        else:
            selected.append(found[lab])
    if missing:
        print(f"[Rank {rank}] Warning: didn't find labels {missing} in val set.")
    return selected  # list of sample dicts in the same order as labels_to_find but missing ones skipped

vis_samples = None
if rank == 0:
    vis_samples = select_val_vis_samples(val_ds, labels_to_find=[1,2,3,4])
    
train_sampler = DistributedSampler(
        train_ds,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed
    )

val_sampler = DistributedSampler(
        val_ds,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=seed
    )

loader_config = {
    "batch_size": bsz,
    "num_workers": 2,
    "pin_memory": False,
    "persistent_workers": True,
    "drop_last": True 
}

train_loader = DataLoader(train_ds, **loader_config, sampler=train_sampler)
val_loader = DataLoader(val_ds, **loader_config, sampler=val_sampler)

## Optimizer & Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(0.02 * len(train_loader) * epoch),
            num_training_steps=len(train_loader) * epoch,
            num_cycles=0.5
        )
noise_scheduler = DDIMScheduler(
            num_train_timesteps=T,
            beta_schedule="squaredcos_cap_v2",
            thresholding=False,
            )

##  Training Loop
scaler = torch.amp.GradScaler('cuda')
best_val_loss = float('inf')
update_ema(ema, model.module, decay=0)  
model.train() 
ema.eval()

print("start training...")
for epoch_i in range(1, epoch + 1):
    train_sampler.set_epoch(epoch_i)
    start_time = time()
    model.train() 
    train_loss = 0.0
    dist.barrier()
    for batch in train_loader:
        x_clean = batch["gt"].to(device)  
        x_cond = batch["cond"].to(device)
        class_labels = batch["label"].to(device) 

        with torch.no_grad():
            z_clean = vae.encode_stage_2_inputs(x_clean, quantized=True)
            z_cond = vae.encode_stage_2_inputs(x_cond, quantized=True)
            
        timesteps = torch.randint(0, T, (bsz,), device=device)
        noise = torch.randn_like(z_clean)
        noisy_images = noise_scheduler.add_noise(z_clean, noise, timesteps)
        model_input = torch.cat([noisy_images, z_cond], dim=1)  

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        #with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            noise_pred = model(model_input, timestep=timesteps, class_labels=class_labels).sample
            loss = F.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        update_ema(ema, model.module)
        train_loss += loss.item()
        
    train_loss_tensor = torch.tensor([train_loss], device=device)
    dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
    global_train_samples = len(train_loader) * dist.get_world_size()
    avg_train_loss = train_loss_tensor.item() / global_train_samples
    
    dist.barrier()
    model.eval()
    val_loss = 0.0
    
    if epoch_i % 25 == 0 and rank == 0 and vis_samples:    
        with torch.no_grad():
            gt_batch = torch.stack([s["gt"] for s in vis_samples]) 
            cond_batch = torch.stack([s["cond"] for s in vis_samples])       
            vis_labels = torch.stack([s["label"] for s in vis_samples]).long().to(device)
            gt_batch = gt_batch.to(device)
            cond_batch = cond_batch.to(device)
            recon   = denoise_batch(model, vae, noise_scheduler, cond_batch, class_labels=vis_labels, n_steps=100)
            save_dir = os.path.join(model_dir, "val_vis"); os.makedirs(save_dir, exist_ok=True)
            fn = os.path.join(save_dir, f"epoch_{epoch_i}.png")   
            plot_triplets(gt_batch, cond_batch, recon, vis_labels, fn)
            
    with torch.no_grad(): 
        for val_batch in val_loader:
            val_clean = val_batch["gt"].to(device)      
            val_cond = val_batch["cond"].to(device)  
            val_class_labels = val_batch["label"].to(device)
            with torch.no_grad():
                val_z_clean = vae.encode_stage_2_inputs(val_clean, quantized=True)
                val_z_cond = vae.encode_stage_2_inputs(val_cond, quantized=True)
                           
            val_timesteps = torch.randint(0, T, (bsz,), device=device)
            val_noise = torch.randn_like(val_z_clean)
            val_noisy_images = noise_scheduler.add_noise(val_z_clean, val_noise, val_timesteps)
            val_model_input = torch.cat([val_noisy_images, val_z_cond], dim=1)
            val_pred = model(val_model_input, timestep=val_timesteps, class_labels=val_class_labels).sample
            val_loss += F.mse_loss(val_pred, val_noise).item()
    
    val_loss_tensor = torch.tensor([val_loss], device=device)
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    global_val_samples = len(val_loader) * dist.get_world_size()
    avg_val_loss = val_loss_tensor.item() / global_val_samples

    if rank == 0:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch_i,
                "ema": ema.state_dict(),
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, f"{model_dir}/best_loss.pth")
             
        log_data = f"Epoch {epoch_i}/{epoch} | "
        log_data += f"Train Loss: {avg_train_loss:.4f} | "
        log_data += f"Val Loss: {avg_val_loss:.4f} | "
        log_data += f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        
        print(log_data)
        with open(log_path, "a") as f:
            f.write(log_data + "\n")
    
    dist.barrier()
    model.train()