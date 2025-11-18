import torch, os, random
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from ArtifactsRemovalDataset import ArtifactsRemovalDataset
from monai.data import DataLoader
from monai.transforms import Compose, Lambdad, EnsureChannelFirstd, ScaleIntensityRanged, RandAffined
from diffusers import UNet2DModel, DDIMScheduler
from generative.networks.nets import VQVAE
import json

# Set random seed
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def compute_metrics_ddpm(pred, gt):
    batch_size = pred.size(0)
    mse_list, ssim_list, psnr_list = [], [], []
    for i in range(batch_size):
        pred_np = pred[i].cpu().detach().numpy().squeeze()
        gt_np = gt[i].cpu().detach().numpy().squeeze()
        
        if pred_np.ndim != 2 or gt_np.ndim != 2:
            raise ValueError(f"Invalid image dimensions: pred_np {pred_np.shape}, gt_np {gt_np.shape}")
    
        mse = np.mean((pred_np - gt_np) ** 2)
        mse_list.append(mse)
        ssim_val = ssim(pred_np, gt_np, data_range=1.0, win_size=7,
                        channel_axis=None if pred_np.ndim == 2 else -1)
        ssim_list.append(ssim_val)
        psnr_val = psnr(pred_np, gt_np, data_range=1.0)
        psnr_list.append(psnr_val)
    return np.mean(mse_list), np.mean(ssim_list), np.mean(psnr_list)


def visualize_results(epoch, gt_imgs, cond_imgs,  pred_imgs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(2, len(cond_imgs))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    titles = ['Ground Truth', 'Condition (with artifacts)', 'Reconstructed (Denoised)']
    for i in range(num_samples):
        cond = cond_imgs[i].cpu().numpy().squeeze()
        gt = gt_imgs[i].cpu().numpy().squeeze()
        pred = pred_imgs[i].cpu().numpy().squeeze()

        axes[i][0].imshow(gt, cmap='gray')
        axes[i][0].set_title(titles[0])
        axes[i][0].axis('off')

        axes[i][1].imshow(cond, cmap='gray')
        axes[i][1].set_title(titles[1])
        axes[i][1].axis('off')

        axes[i][2].imshow(pred, cmap='gray')
        axes[i][2].set_title(titles[2])
        axes[i][2].axis('off')

        psnr_val = psnr(gt, pred, data_range=1.0)
        ssim_val = ssim(gt, pred, data_range=1.0, win_size=7)
    
        metrics_text = f'PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.3f}'
        axes[i][2].text(1.05, 0.5, metrics_text,
                        transform=axes[i][2].transAxes,
                        verticalalignment='center',
                        fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_inference_metrics_to_txt(metrics, save_path):
    try:
        with open(save_path, 'w') as f:
            f.write("Test_MSE\tTest_SSIM\tTest_PSNR\n")
            f.write(f"{metrics['mse']:.6f}\t"
                    f"{metrics['ssim']:.4f}\t"
                    f"{metrics['psnr']:.4f}\n")
        print(f"Inference metrics successfully saved to: {save_path}")
    except Exception as e:
        print(f"Failed to save inference metrics to {save_path}, error: {e}")


def generate_denoised_samples(model, vae, noise_scheduler, cond_batch, num_inference_steps=100):
    z_cond = vae.encode_stage_2_inputs(cond_batch, quantized=True)
    denoised_image_samples = torch.randn_like(z_cond)
    noise_scheduler.set_timesteps(num_inference_steps)

    for t in noise_scheduler.timesteps:
        model_input = torch.cat([denoised_image_samples, z_cond], dim=1)
        noise_pred = model(model_input, t).sample
        denoised_image_samples = noise_scheduler.step(noise_pred, t, denoised_image_samples).prev_sample
    
    with torch.no_grad():   
        denoise_images = vae.decode(denoised_image_samples)
        
    return denoise_images



def main(train_flag=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("./vqvae_lat1/vqvae_config.json", "r") as f:
        config = json.load(f)
    vae = VQVAE(**config)
    vae.load_state_dict(torch.load("./vqvae_lat1/best_vqvae.pth", map_location=device))
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()
    data_dir = 'data'
    train_cond_dir = os.path.join(data_dir, 'train', 'motion_blur') 
    train_gt_dir = os.path.join(data_dir, 'train', 'gt')      
    test_cond_dir = os.path.join(data_dir, 'test', 'motion_blur') 
    test_gt_dir = os.path.join(data_dir, 'test', 'gt') 

    output_dir = 'ddpm_motion_blur_lat1/' 
    epochs = 500
    batch_size = 8
    lr = 1e-4
    num_train_timesteps_scheduler = 1000
    num_inference_steps_sampler = 100
    best_val_loss = float('inf')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'inference'), exist_ok=True)
  
    transform = Compose(
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
    
    
    full_dataset = ArtifactsRemovalDataset(train_gt_dir, train_cond_dir, transform)                                                
    test_dataset = ArtifactsRemovalDataset(test_gt_dir, test_cond_dir, transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size],
                                                     generator=torch.Generator().manual_seed(seed)
)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Unet
    model = UNet2DModel(
        sample_size=64,
        in_channels=2, 
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"
        ),
        up_block_types=(
            "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
    ).to(device)
    
    noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps_scheduler,
            beta_schedule="squaredcos_cap_v2",
            thresholding=False,
            )
    if train_flag:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_idx, data_batch in enumerate(train_loader):
                batch_cond_imgs, batch_gt_imgs = data_batch["cond"].to(device), data_batch["gt"].to(device)
                with torch.no_grad():
                      z_clean = vae.encode_stage_2_inputs(batch_gt_imgs, quantized=True)
                      z_cond = vae.encode_stage_2_inputs(batch_cond_imgs, quantized=True)
            
                noise_added = torch.randn_like(z_clean)
                timesteps_tensor = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                                 (z_clean.shape[0],), device=device)
                noisy_gt_batch = noise_scheduler.add_noise(z_clean, noise_added, timesteps_tensor)
                unet_input = torch.cat([noisy_gt_batch, z_cond], dim=1)
                predicted_noise = model(unet_input, timesteps_tensor).sample

                optimizer.zero_grad()
                loss = F.mse_loss(predicted_noise, noise_added)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"[Epoch {epoch + 1}/{epochs}] Avg Train Loss: {avg_train_loss:.6f}")

            if (epoch + 1) % 25 == 0:
                model.eval()
                with torch.no_grad():
                    val_batch = next(iter(val_loader))
                    cond_imgs = val_batch["cond"].to(device)
                    gt_imgs = val_batch["gt"].to(device)
                    pred_imgs = generate_denoised_samples(
                            model, vae, noise_scheduler, cond_imgs,
                            num_inference_steps=num_inference_steps_sampler
                    )
                    visualize_results(epoch + 1, gt_imgs, cond_imgs, pred_imgs,
                                          os.path.join(output_dir, 'visualizations'))
                        
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    cond_val, gt_val = val_batch["cond"].to(device), val_batch["gt"].to(device)
                    with torch.no_grad():
                        val_z_clean = vae.encode_stage_2_inputs(gt_val, quantized=True)
                        val_z_cond = vae.encode_stage_2_inputs(cond_val, quantized=True)
                    noise_val = torch.randn_like(val_z_clean)
                    timesteps_tensor_val = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                                 (val_z_clean.shape[0],), device=device)
                    noisy_gt_val = noise_scheduler.add_noise(val_z_clean, noise_val, timesteps_tensor_val)

                    unet_val_input = torch.cat([noisy_gt_val, val_z_cond], dim=1)
                    predicted_val_noise = model(unet_val_input, timesteps_tensor_val).sample
                    val_batch_loss = F.mse_loss(predicted_val_noise, noise_val)
                    val_loss += val_batch_loss.item()

                avg_val_loss = val_loss / len(val_loader)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(output_dir, 'models', 'best_model.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss
                    }, best_model_path)
                    print(f"Best model updated at epoch {epoch + 1}, saved to {best_model_path}")
                   

    else:
        path_model_to_load = os.path.join(output_dir, 'models', 'best_model.pth')
        checkpoint = torch.load(path_model_to_load, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Loaded best model from epoch {checkpoint['epoch']} with val loss {checkpoint['best_val_loss']:.6f}")

        test_mse, test_ssim, test_psnr = [], [], []

        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                cond_imgs = batch["cond"].to(device)
                gt_imgs = batch["gt"].to(device)

                pred_imgs = generate_denoised_samples(model, vae, noise_scheduler, cond_imgs)

                mse, ssim_val, psnr_val = compute_metrics_ddpm(pred_imgs, gt_imgs)
                test_mse.append(mse)
                test_ssim.append(ssim_val)
                test_psnr.append(psnr_val)

           
            if idx == 0:
                visualize_results("test", cond_imgs, gt_imgs, pred_imgs, os.path.join(output_dir, 'inference'))

    
        final_metrics = {
            'mse': np.mean(test_mse),
            'ssim': np.mean(test_ssim),
            'psnr': np.mean(test_psnr)
        }

        print(f"Test Set Metrics:")
        print(f"MSE:  {final_metrics['mse']:.6f}")
        print(f"SSIM: {final_metrics['ssim']:.4f}")
        print(f"PSNR: {final_metrics['psnr']:.4f}")

        save_inference_metrics_to_txt(final_metrics, os.path.join(output_dir, 'inference', 'metrics.txt')) 

        
if __name__ == "__main__":
    main(train_flag=True)