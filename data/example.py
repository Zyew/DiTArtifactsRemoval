import math
import numpy as np
import torch, os
import matplotlib.pyplot as plt
from diffct.differentiable import ConeProjectorFunction, ConeBackprojectorFunction
import pydicom
import scipy.ndimage as ndi
import gc
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def dicom_volume(dicom_folder):
    dicom_files = [
        os.path.join(dicom_folder, f)
        for f in os.listdir(dicom_folder)
        if f.lower().endswith(".dcm")
    ]
    if not dicom_files:
        raise FileNotFoundError(f"Not found dicom files：{dicom_folder}")
    
    dicom_files.sort()
    slices = []
    ds0 = pydicom.dcmread(dicom_files[0])
    center = ds0.WindowCenter
    width = ds0.WindowWidth
    vmin = center - width / 2
    vmax = center + width / 2
    
    for f in dicom_files:
        ds = pydicom.dcmread(f)
        instance = getattr(ds, "InstanceNumber", 0)
        img = ds.pixel_array
        slope = getattr(ds, 'RescaleSlope', 1.0)
        intercept = getattr(ds, 'RescaleIntercept', 0.0)
        hu_img = img * slope + intercept
        hu_img = hu_img.astype(np.float32)
        hu_img_clipped = np.clip(hu_img, vmin, vmax).astype(np.float32)
        hu_img_normalized = (hu_img_clipped - vmin) / (vmax - vmin)
        slices.append((instance, hu_img_normalized))

    slices.sort(key=lambda x: x[0])
    vol = np.stack([s[1] for s in slices], axis=2)
    return vol

def add_motion(sinogram, max_jitter, max_rotation, max_scale):
        """
        Apply translation, rotation, and scaling to the sinogram to simulate motion artifacts.

        :param max_jitter: Maximum amount of jitter (in pixels)
        :param max_rotation: Maximum rotation angle (in degrees)
        :param max_scale: Maximum scaling factor
        """
        num_projections, H, W = sinogram.shape
        motion_sinogram = np.zeros_like(sinogram)

        for i in range(num_projections):
            projection = sinogram[i, :, :]

            # translation
            jitter = np.random.randint(-max_jitter, max_jitter)
            projection_after_translation = np.roll(projection, jitter, axis=1)
            projection_after_translation = np.roll(projection_after_translation, jitter, axis=0)
            
            # rotation
            angle = np.random.uniform(-max_rotation, max_rotation)
            rotated = ndi.rotate(projection, angle=angle, reshape=False, order=1, mode='reflect')
        
            # scaling
            scale = 1.0 + np.random.uniform(-max_scale, max_scale)
            scaled = ndi.zoom(rotated, zoom=scale, order=1)
            h_new, w_new = scaled.shape

            # --- Crop if scaled projection is larger ---
            if h_new > H:
                start_h = (h_new - H) // 2
                end_h = start_h + H
                scaled = scaled[start_h:end_h, :]
            elif h_new < H:
                pad_h = H - h_new
                scaled = np.pad(scaled, ((pad_h // 2, pad_h - pad_h // 2), (0, 0)), mode='reflect')

            if w_new > W:
                start_w = (w_new - W) // 2
                end_w = start_w + W
                scaled = scaled[:, start_w:end_w]
            elif w_new < W:
                pad_w = W - w_new
                scaled = np.pad(scaled, ((0, 0), (pad_w // 2, pad_w - pad_w // 2)), mode='reflect')

            motion_sinogram[i] = scaled

        return motion_sinogram
           
        

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



def main():

    input_folder = "./1/CT Plain"
    phantom_cpu = dicom_volume(input_folder)
    Nx, Ny, Nz = phantom_cpu.shape

    num_views = 720
    angles_np = np.linspace(0, 2*math.pi, num_views, endpoint=False).astype(np.float32)

    det_u, det_v = 1024, 1024
    du, dv = 1.0, 1.0
    source_distance = 1200.0
    isocenter_distance = 800.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom_torch = torch.tensor(phantom_cpu, device=device, requires_grad=True)
    angles_torch = torch.tensor(angles_np, device=device)

    sinogram = ConeProjectorFunction.apply(phantom_torch, angles_torch,
                                           det_u, det_v, du, dv,
                                           source_distance, isocenter_distance)
    
    sinogram_motion_np = add_motion(sinogram.detach().cpu().numpy(),
                              max_jitter=1,
                              max_rotation=0.1,
                              max_scale=0.005)
    sinogram_motion = torch.tensor(sinogram_motion_np, dtype=phantom_torch.dtype, device=device)
    torch.cuda.empty_cache()
    def fdk_reconstruct(sinogram_tensor):
        u_coords = (torch.arange(det_u, dtype=phantom_torch.dtype, device=device) - (det_u - 1) / 2) * du
        v_coords = (torch.arange(det_v, dtype=phantom_torch.dtype, device=device) - (det_v - 1) / 2) * dv
        u_coords = u_coords.view(1, det_u, 1)
        v_coords = v_coords.view(1, 1, det_v)
        weights = source_distance / torch.sqrt(source_distance**2 + u_coords**2 + v_coords**2)
        sino_weighted = sinogram_tensor * weights
        sino_filt = ramp_filter_3d(sino_weighted).detach().requires_grad_(True).contiguous()
        
        recon = ConeBackprojectorFunction.apply(sino_filt, angles_torch, Nx, Ny, Nz,
                                                du, dv, source_distance, isocenter_distance)
        del sino_filt
        gc.collect()
        torch.cuda.empty_cache()
        return recon * (math.pi / num_views) / (du * dv)
    
    reconstruction_nomotion = fdk_reconstruct(sinogram)
    reconstruction_motion = fdk_reconstruct(sinogram_motion)

    phantom_cpu = phantom_torch.detach().cpu().numpy()
    recon_nomotion_cpu = reconstruction_nomotion.detach().cpu().numpy()
    recon_motion_cpu = reconstruction_motion.detach().cpu().numpy()
    mid_slice = Nz // 2

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(phantom_cpu[:,:,mid_slice], cmap='gray')
    plt.title("Phantom mid-slice")
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(recon_nomotion_cpu[:,:,mid_slice], cmap='gray')
    plt.title("Recon (no motion)")
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(recon_motion_cpu[:,:,mid_slice], cmap='gray')
    plt.title("Recon (motion)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("fdk_cone_motion_compare_tryall_norm0_1.png", dpi=300)
    plt.show()

    # print data range of the phantom and reco
    #print("Loss (motion vs GT):", loss.item())
    #print("Volume center voxel gradient:", phantom_torch.grad[Nx//2, Ny//2, Nz//2].item())
    print("Phantom range:", phantom_cpu.min(), phantom_cpu.max())
    print("Reco (no motion) range:", recon_nomotion_cpu.min(), recon_nomotion_cpu.max())
    print("Reco (motion) range:", recon_motion_cpu.min(), recon_motion_cpu.max())
if __name__ == "__main__":
    main()