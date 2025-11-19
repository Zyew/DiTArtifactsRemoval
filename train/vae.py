import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import Compose, Lambdad, EnsureChannelFirstd, ScaleIntensityRanged, RandAffined
from CTDataset import CTDataset
from monai.data import DataLoader
from monai.utils import set_determinism
from torch.nn import L1Loss
from tqdm import tqdm
from generative.networks.nets import VQVAE
from torchvision.utils import save_image
import json
import glob
from piq import LPIPS

set_determinism(2025)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def visualize_fixed_slices(model, root_dir, save_path, device, use_middle_slice=True):
    model.eval()
    model.to(device)

    transform = Compose([
        Lambdad(keys=["image"], func=lambda x: x[None, ...]),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True),
    ])

    input_images = []
    titles = []

    for organ in sorted(os.listdir(root_dir)):
        organ_train_path = os.path.join(root_dir, organ)
        if not os.path.isdir(organ_train_path):
            continue

        for subfolder in sorted(os.listdir(organ_train_path)):
            subfolder_path = os.path.join(organ_train_path, subfolder)
            npy_path = os.path.join(subfolder_path, "1.npy")

            if not os.path.isfile(npy_path):
                continue 

            try:
                volume = np.load(npy_path)
                if volume.ndim == 4:
                    volume = volume[0]

                slice_idx = volume.shape[0] // 2 if use_middle_slice else 0
                slice_img = volume[slice_idx]

                sample = {"image": slice_img}
                sample = transform(sample)
                input_images.append(sample["image"])
                titles.append(f"{organ}/{subfolder}")

            except Exception as e:
                print(f"[Warning] Failed loading {npy_path}: {e}")


        if len(input_images) == 0:
            print("No valid input images found.")
            return
    
    input_tensor = torch.stack(input_images).to(device)
    with torch.no_grad():
        recon_tensor, _ = model(input_tensor)

    input_tensor = input_tensor.cpu().numpy()
    recon_tensor = recon_tensor.cpu().numpy()
    n = len(input_tensor)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for i in range(n):
        axes[0, i].imshow(input_tensor[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"{titles[i]}", fontsize=10)

        axes[1, i].imshow(recon_tensor[i, 0], cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Input", fontsize=12)
            axes[1, i].set_ylabel("Recon", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Data setup
image_size = 512
batch_size = 16
root_dir = "/home/hpc/iwi5/iwi5220h/diffct/data/train"
val_interval = 5
n_epochs = 200
save_dir = "/root_dir/train/vqvae_with_perceptual"
os.makedirs(save_dir, exist_ok=True)
epoch_recon_loss_list = []
epoch_quant_loss_list = []
val_recon_epoch_loss_list = []

train_transforms = Compose(
    [
        Lambdad(keys=["image"], func=lambda x: x[None, ...]),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True),
        RandAffined(
            keys=["image"],
            spatial_size=[512, 512],
            prob=0.5 
        ),
    ]   
    )

dataset = CTDataset(root_dir, transform=train_transforms)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

# Model
config = {
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1,
    "num_res_layers": 3,
    "downsample_parameters": [[2, 4, 1, 1], [2, 4, 1, 1], [2, 4, 1, 1]],
    "upsample_parameters": [[2, 4, 1, 1, 0], [2, 4, 1, 1, 0], [2, 4, 1, 1, 0]],
    "num_channels": [256, 512, 512],
    "num_res_channels": [256, 512, 512],
    "num_embeddings": 8192,
    "embedding_dim": 1,
}

with open(os.path.join(save_dir, "vqvae_config.json"), "w") as f:
    json.dump(config, f, indent=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE(**config).to(device)
lpips_loss = LPIPS(reduction='mean').to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
l1_loss = L1Loss()

# Training
total_start = time.time()
best_val_loss = float("inf")
kl_weight = 0.01
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, quantization_loss = model(images)
        recons_loss = l1_loss(reconstruction.float(), images.float())
        image_repeat = images.repeat(1, 3, 1, 1) * 2 - 1
        recon_repeat = reconstruction.repeat(1, 3, 1, 1) * 2 - 1
        percep_loss = lpips_loss(recon_repeat, image_repeat)
        loss = recons_loss + quantization_loss + 0.2 * percep_loss
        #loss = recons_loss + quantization_loss
        loss.backward()
        optimizer.step()
        epoch_loss += recons_loss.item()
        progress_bar.set_postfix(
            {"recons_loss": epoch_loss / (step + 1), "quantization_loss": quantization_loss.item() / (step + 1)}
        )
     
    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_quant_loss_list.append(quantization_loss.item() / (step + 1))

    if (epoch + 1) % val_interval == 0:
        model.eval()

        visualize_fixed_slices(
        model=model,
        root_dir=root_dir,
        save_path=os.path.join(save_dir, f"vis_epoch{epoch+1}.png"),
        device=device,
        use_middle_slice=True
        )

        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)
                reconstruction, quantization_loss = model(images=images)
                recons_loss = l1_loss(reconstruction.float(), images.float())

                val_loss += recons_loss.item()

        val_loss /= len(val_loader)
        val_recon_epoch_loss_list.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_vqvae.pth"))
         
# Save final model and visualization
torch.save(model.state_dict(), os.path.join(save_dir, "final_vqvae.pth"))
visualize_fixed_slices(
    model=model,
    root_dir=root_dir,
    save_path=os.path.join(save_dir, "final_reconstruction.png"),
    device=device,
    use_middle_slice=True
)

# Visualize best model
best_model_path = os.path.join(save_dir, "best_vqvae.pth")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    visualize_fixed_slices(
        model=model,
        root_dir=root_dir,
        save_path=os.path.join(save_dir, "best_reconstruction.png"),
        device=device,
        use_middle_slice=True
    )

# Plot learning curves
plt.style.use("ggplot")
plt.title("Learning Curves", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_recon_loss_list, color="C0", linewidth=2.0, label="Train")
plt.plot(
    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
    val_recon_epoch_loss_list,
    color="C1",
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig(os.path.join(save_dir, "loss_curve.png"))
plt.close()
