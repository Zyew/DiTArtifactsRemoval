from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
from diffusers import DDIMScheduler
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch, random
from diffusers import DDIMScheduler
from ArtifactsRemovalDataset import ArtifactsRemovalDataset
from generative.networks.nets import VQVAE
from monai.transforms import Compose, Lambdad, EnsureChannelFirstd, ScaleIntensityRanged, RandAffined
import json
from monai.data import DataLoader

#
parser = ArgumentParser()
parser.add_argument("--train_timesteps_number", type=int, default=1000)
parser.add_argument("--num_inference_steps", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--patch_size", type=int, default=4)
args = parser.parse_args()

seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VAE
with open("./vqvae_4/vqvae_config.json", "r") as f:
    vae_config = json.load(f)

vae = VQVAE(**vae_config)
vae.load_state_dict(torch.load("./vqvae_4/best_vqvae.pth", map_location=device))
vae = vae.to(device)
vae.requires_grad_(False)
vae.eval()

# Load DiT
with open("./checkpoints/dit_detector_jitter/dit_config.json", "r") as f:
    dit_config = json.load(f)

model = DiTTransformer2DModel(**dit_config).to(device)
model_path = "./checkpoints/dit_detector_jitter/ep1000_bsz8_0618_1215/best_loss.pth"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.train_timesteps_number,
            beta_schedule="squaredcos_cap_v2",
            thresholding=False,
            )

def _denoise_image(condition: torch.Tensor) -> torch.Tensor:
    B = condition.shape[0]
    z_cond = vae.encode_stage_2_inputs(condition, quantized=True)
    noisy_image = torch.randn_like(z_cond)
   
    for t in noise_scheduler.timesteps:
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)    
        model_input = torch.cat([noisy_image, z_cond], dim=1)
        class_labels = 1000 * torch.ones(B, dtype=torch.long, device=device)      
        noise_pred = model(model_input, t_batch, class_labels).sample
        noisy_image = noise_scheduler.step(noise_pred, t, noisy_image).prev_sample
    
    with torch.no_grad():   
   
        denoise_images = vae.decode(noisy_image)
        
    return denoise_images


def _plot_results(clean_images: torch.Tensor, conditions: torch.Tensor, results: torch.Tensor,
                      num_samples: int = 4, save_path: str = 'restoration_results_bh.png'):
    """
    Plot multiple results in a grid layout and save as a single image.

    Args:
        clean_images: Original clean images
        conditions: Noisy condition images
        results: Generated restoration results
        num_samples: Number of samples to plot
        save_path: Path to save the result image
    """
  
    # Create a grid of plots: num_samples rows x 3 columns
    _, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    # Set column titles
    titles = ['Ground Truth', 'Condition', 'Generated Image']
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=12, pad=10)

    # Plot each set of images
    for idx in range(num_samples):
        images = [
            clean_images[idx:idx+1],
            conditions[idx:idx+1],
            results[idx:idx+1]
        ]

        for col, img in enumerate(images):
            axes[idx, col].imshow(img[0, 0].cpu().T, cmap='gray')
            axes[idx, col].axis('off')

            # Add PSNR and SSIM metrics for generated images
            if col == 2:  # Generated image column
                psnr = _calculate_psnr(clean_images[idx:idx+1], img)
                ssim = _calculate_ssim(clean_images[idx:idx+1], img)
                metrics_text = f'PSNR: {psnr:.2f}\nSSIM: {ssim:.3f}'
                axes[idx, col].text(1.05, 0.5, metrics_text,
                                    transform=axes[idx, col].transAxes,
                                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def _calculate_psnr(clean_img: torch.Tensor, generated_img: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = torch.mean((clean_img.cpu() - generated_img.cpu()) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def _calculate_ssim(clean_img: torch.Tensor, generated_img: torch.Tensor) -> float:
    """
    Calculate Structural Similarity Index between two images.
    This is a simplified version of SSIM.
    """
    clean_img = clean_img.float()
    generated_img = generated_img.float()

    clean_img = clean_img.cpu()
    generated_img = generated_img.cpu()

    # Constants for stability
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2

    # Calculate means
    mu1 = torch.mean(clean_img)
    mu2 = torch.mean(generated_img)

    # Calculate variances and covariance
    sigma1_sq = torch.var(clean_img)
    sigma2_sq = torch.var(generated_img)
    sigma12 = torch.mean((clean_img - mu1) * (generated_img - mu2))

    # Calculate SSIM
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim.item()

train_transforms = Compose(
    [
        Lambdad(keys=["gt", "cond"], func=lambda x: x[None, ...]),
        ScaleIntensityRanged(keys=["gt", "cond"], a_min=0.0, a_max=4095.0, b_min=0.0, b_max=1.0, clip=True),
        RandAffined(
            keys=["gt", "cond"],
            spatial_size=[512, 512],
            prob=0.5 
        ),
    ]   
    )

test_dataset = ArtifactsRemovalDataset("./data/test/gt",
                                  "./data/test/detector_jitter", 
                                  transform=train_transforms)
test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True
)
noise_scheduler.set_timesteps(args.num_inference_steps)

with torch.no_grad():
    batch = next(iter(test_dataloader))
    gt = batch["gt"].to(device)  
    cond = batch["cond"].to(device)
    recon = _denoise_image(cond)
    num_samples = min(gt.shape[0], 2)
    _plot_results(gt[:num_samples], cond[:num_samples], recon[:num_samples], 
                  num_samples=num_samples, save_path='restoration_results_bh.png')