import math
import numpy as np
import torch, os
import matplotlib.pyplot as plt
from diffct.differentiable import ConeProjectorFunction, ConeBackprojectorFunction
import gc
from tqdm import tqdm
from scipy.ndimage import convolve, gaussian_filter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

input_folder = "./data/pancreas/train/gt"
output_base = "./data/pancreas/train_artifacts"
normalize_divisor = 4071.0 
num_views = 720
det_u, det_v = 1024, 1024
du, dv = 1.0, 1.0
source_distance = 1200.0
isocenter_distance = 800.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
angles_np = np.linspace(0, 2 * math.pi, num_views, endpoint=False).astype(np.float32)
angles_torch = torch.tensor(angles_np, device=device)


def load_npy_volume(npy_path):
    volume = np.load(npy_path).astype(np.float32)
    assert volume.min() >= 0 and volume.max() <= normalize_divisor, "Unexpected value range!"
    volume = np.transpose(volume, (1, 2, 0))
    volume_norm = volume.copy() / normalize_divisor
    print(f"[INFO] Normalized volume range: min={volume_norm.min()}, max={volume_norm.max()}")
    return volume_norm

def normalize(volume):
    vmin, vmax = volume.min(), volume.max()
    return np.clip((volume - vmin) / (vmax - vmin + 1e-8), 0, 1)

def add_detector_jitter(sinogram, max_jitter=5):
    """
    Simulate detector jitter in the sinogram.

    :param max_jitter: Maximum amount of jitter (in pixels)
    """
    num_projections = sinogram.shape[0]
    jittered_sinogram = np.zeros_like(sinogram)

    for i in range(num_projections):
        jitter = np.random.randint(-max_jitter, max_jitter)
        jittered_projection = np.roll(sinogram[i, :, :], jitter, axis=1)
        jittered_projection = np.roll(jittered_projection, jitter, axis=0)
        jittered_sinogram[i, :, :] = jittered_projection

    return jittered_sinogram

def add_high_frequency_noise(sinogram, noise_level=0.5, high_freq_strength=2.0):
    """
    Add high-frequency noise to the sinogram.
    :param noise_level: Standard deviation of the Gaussian noise
    :param high_freq_strength: Strength of the high-frequency component
    """
    num_projections = sinogram.shape[0]
    noisy_sinogram = np.zeros_like(sinogram)

    for i in range(num_projections):
        noise = np.random.normal(0, noise_level, size=sinogram.shape[1:])
        low_freq_noise = gaussian_filter(noise, sigma=high_freq_strength)
        high_freq_noise = noise - low_freq_noise
        noisy_sinogram[i, :, :] = sinogram[i, :, :] + high_freq_noise * high_freq_strength
    return noisy_sinogram

def add_poisson_noise(sinogram, scale_factor=50.0):
    """
    Add Poisson noise to the sinogram.
    :param scale_factor: Scale factor to control the intensity of the noise
    """
    noisy_sinogram = np.zeros_like(sinogram)
    for i in range(sinogram.shape[0]):
        noisy_sinogram[i, :, :] = np.random.poisson(sinogram[i, :, :] * scale_factor) / scale_factor
   
    return noisy_sinogram

def apply_artifact(sino_tensor, artifact_type):
    sino_np = sino_tensor.detach().cpu().numpy()
    if artifact_type == "poisson":
        sino_np = add_poisson_noise(sino_np)
    elif artifact_type == "hf":
        sino_np = add_high_frequency_noise(sino_np)
    elif artifact_type == "jitter":
        sino_np = add_detector_jitter(sino_np)
    return torch.tensor(sino_np, dtype=sino_tensor.dtype, device=sino_tensor.device)


def ramp_filter_3d(sinogram_tensor):
    device = sinogram_tensor.device
    num_views, num_det_u, num_det_v = sinogram_tensor.shape
    freqs = torch.fft.fftfreq(num_det_u, device=device)
    omega = 2.0 * torch.pi * freqs
    ramp = torch.abs(omega)
    ramp_3d = ramp.reshape(1, num_det_u, 1)
    sino_fft = torch.fft.fft(sinogram_tensor, dim=1)
    filtered_fft = sino_fft * ramp_3d
    filtered = torch.real(torch.fft.ifft(filtered_fft, dim=1))
    
    return filtered


def fdk_reconstruct(sinogram_tensor, Nx, Ny, Nz):
    u = (torch.arange(det_u, dtype=sinogram_tensor.dtype, device=device) - (det_u - 1) / 2) * du
    v = (torch.arange(det_v, dtype=sinogram_tensor.dtype, device=device) - (det_v - 1) / 2) * dv
    u = u.view(1, det_u, 1)
    v = v.view(1, 1, det_v)
    weights = source_distance / torch.sqrt(source_distance**2 + u**2 + v**2)
    sino_weighted = sinogram_tensor * weights
    sino_filt = ramp_filter_3d(sino_weighted).contiguous()
    with torch.no_grad():
        recon = ConeBackprojectorFunction.apply(
            sino_filt, angles_torch, Nx, Ny, Nz,
            du, dv, source_distance, isocenter_distance
        )

    del sino_filt
    gc.collect()
    torch.cuda.empty_cache()

    return recon * (math.pi / num_views) / (du * dv)

def process_all():
    artifact_types = ["jitter", "poisson", "hf"]
    os.makedirs(output_base, exist_ok=True)

    
    for sub in ["gt", "vis"] + artifact_types:
        os.makedirs(os.path.join(output_base, sub), exist_ok=True)

    file_list = sorted([f for f in os.listdir(input_folder) if f.endswith('.npy')])

    for idx, fname in enumerate(tqdm(file_list)):
        npy_path = os.path.join(input_folder, fname)
        phantom_cpu = load_npy_volume(npy_path)
        Nx, Ny, Nz = phantom_cpu.shape

        phantom_torch = torch.tensor(phantom_cpu, device=device)
        with torch.no_grad():
            sinogram = ConeProjectorFunction.apply(
                phantom_torch, angles_torch,
                det_u, det_v, du, dv,
                source_distance, isocenter_distance
        )
      
        phantom_np = phantom_torch.detach().cpu().numpy()
        print(f"[GT] {fname} | min={phantom_np.min():.4f}, max={phantom_np.max():.4f}")
        np.save(os.path.join(output_base, "gt", f"{idx + 1}.npy"),
                np.transpose(phantom_np, (2, 0, 1)).astype(np.float32))

        
        for art_type in artifact_types:
            sino_art = apply_artifact(sinogram, art_type)
            recon_art = fdk_reconstruct(sino_art, Nx, Ny, Nz)
            recon_art_np = recon_art.detach().cpu().numpy()
            recon_art_norm = normalize(recon_art_np)

            np.save(os.path.join(output_base, art_type, f"{idx + 1}.npy"),
                    np.transpose(recon_art_norm, (2, 0, 1)).astype(np.float32))
            print(f"[{art_type.upper()}] {fname} |"
                  f"min={recon_art_norm.min():.4f}, max={recon_art_norm.max():.4f}")

            
            plt.figure(figsize=(12, 4))
            mid = Nz // 2
            plt.subplot(1, 2, 1)
            plt.imshow(phantom_np[:, :, mid], cmap='gray')
            plt.title("GT"); plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(recon_art_norm[:, :, mid], cmap='gray')
            plt.title(f"{art_type}"); plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_base, "vis", f"{idx + 1}_{art_type}.png"))
            plt.close()

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    process_all()
