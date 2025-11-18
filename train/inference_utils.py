import torch, matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from diffusers import DDIMScheduler


noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            thresholding=False,
            )

def denoise_batch(model, vae, noise_scheduler, cond_batch, n_steps: int = 100):
    noise_scheduler.set_timesteps(n_steps)
    B = cond_batch.shape[0]
    device = cond_batch.device
    z_cond = vae.encode_stage_2_inputs(cond_batch, quantized=True)
    latent = torch.randn_like(z_cond)  

    for t in noise_scheduler.timesteps:
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)
        inp = torch.cat([latent, z_cond], dim=1)
        class_labels = torch.full((B,), 1000, dtype=torch.long, device=device)
        eps = model(inp, t_batch, class_labels).sample
        latent = noise_scheduler.step(eps, t, latent).prev_sample
    return vae.decode(latent).clamp(0., 1.)             

def calc_metrics(clean, recon):
    psnr_val = psnr(clean.cpu().numpy(), recon.cpu().numpy(), data_range=1.0)
    ssim_val = ssim(clean.cpu().numpy(), recon.cpu().numpy(), data_range=1.0)
    return psnr_val, ssim_val

def plot_triplets(gt, cond, recon, fn):
    n = gt.shape[0]
    fig, ax = plt.subplots(n, 3, figsize=(10, 4*n))
    titles = ["GT", "Condition", "Reconstruction"]
    for i in range(n):
        for j, img in enumerate([gt, cond, recon]):
            ax[i, j].imshow(img[i,0].cpu(), cmap='gray')
            ax[i, j].axis('off')
            if i == 0: ax[i, j].set_title(titles[j])
        psnr_v, ssim_v = calc_metrics(gt[i,0], recon[i,0])
        ax[i, 2].text(1.05, 0.5, f'PSNR {psnr_v:.2f}\nSSIM {ssim_v:.3f}',
                      transform=ax[i,2].transAxes, va='center')
    plt.tight_layout(); plt.savefig(fn, dpi=300); plt.close()