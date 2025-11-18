import math
import torch, os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from diffct.differentiable import ConeProjectorFunction
import pydicom
import scipy.ndimage as ndi


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
        #hu_clip = np.clip(img, vmin, vmax).astype(np.float32)
        slices.append((instance, hu_img_clipped))

    slices.sort(key=lambda x: x[0])
    vol = np.stack([s[1] for s in slices], axis=0)
    return vol


class IterativeRecoModel(nn.Module):
    def __init__(self, volume_shape, angles, det_u, det_v, du, dv, source_distance, isocenter_distance):
        super().__init__()
        self.reco = nn.Parameter(torch.zeros(volume_shape))
        self.angles = angles
        self.det_u = det_u
        self.det_v = det_v
        self.du = du
        self.dv = dv
        self.source_distance = source_distance
        self.isocenter_distance = isocenter_distance

    def forward(self, x):
        updated_reco = x + self.reco
        current_sino = ConeProjectorFunction.apply(updated_reco, 
                                                   self.angles, 
                                                   self.det_u, self.det_v, 
                                                   self.du, self.dv, 
                                                   self.source_distance, self.isocenter_distance)
        return current_sino, updated_reco

class Pipeline:
    def __init__(self, lr, volume_shape, angles, 
                 det_u, det_v, du, dv, 
                 source_distance, isocenter_distance, 
                 device, epoches=1000):
        
        self.epoches = epoches
        self.model = IterativeRecoModel(volume_shape, angles,
                                        det_u, det_v, du, dv, 
                                        source_distance, isocenter_distance).to(device)
        
        self.optimizer = optim.AdamW(list(self.model.parameters()), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5) #
        self.loss = nn.MSELoss()

    def train(self, input, label):
        loss_values = []
        for epoch in range(self.epoches):
            self.optimizer.zero_grad()
            predictions, current_reco = self.model(input)
            loss_value = self.loss(predictions, label)
            loss_value.backward()
            self.optimizer.step()
            self.scheduler.step() #
            loss_values.append(loss_value.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value.item()}")

        return loss_values, self.model

def main():
    phantom_cpu = dicom_volume("./1-50/1/CT Plain")
    Nz, Ny, Nx = phantom_cpu.shape
    num_views = 180
    angles_np = np.linspace(0, 2 * math.pi, num_views, endpoint=False).astype(np.float32)

    det_u, det_v = 1280, 1280
    du, dv = 0.75, 0.75
    source_distance = 1200.0
    isocenter_distance = 800.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom_torch = torch.tensor(phantom_cpu, device=device)
    angles_torch = torch.tensor(angles_np, device=device)

    # Generate the "real" sinogram
    real_sinogram = ConeProjectorFunction.apply(phantom_torch, angles_torch,
                                               det_u, det_v, du, dv,
                                               source_distance, isocenter_distance)

    pipeline_instance = Pipeline(lr=1e-2, 
                                 volume_shape=(Nz,Ny,Nx), 
                                 angles=angles_torch, 
                                 det_u=det_u, det_v=det_v, 
                                 du=du, dv=dv, 
                                 source_distance=source_distance, 
                                 isocenter_distance=isocenter_distance, 
                                 device=device, epoches=1000)
    
    ini_guess = torch.zeros_like(phantom_torch)
    
    loss_values, trained_model = pipeline_instance.train(ini_guess, real_sinogram)
    
    reco = trained_model(ini_guess)[1].squeeze().cpu().detach().numpy()

    plt.figure()
    plt.plot(loss_values)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    mid_slice = Nz // 2
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(phantom_cpu[mid_slice, :, :], cmap="gray")
    plt.title("Original Phantom Mid-Slice")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reco[mid_slice, :, :], cmap="gray")
    plt.title("Reconstructed Mid-Slice")
    plt.axis("off")
    plt.savefig("iter_cone_1e-2.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()