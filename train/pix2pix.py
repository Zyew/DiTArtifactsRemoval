import torch, os, random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from monai.data import DataLoader
from monai.transforms import Compose, Lambdad, EnsureChannelFirstd, ScaleIntensityRanged, RandAffined
from PairedInputDataset import PairedInputDataset


# Set random seed
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetGenerator, self).__init__()
        def down_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
        def up_block(in_ch, out_ch, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)
        self.enc1 = down_block(in_channels, 64)
        self.enc2 = down_block(64, 128)
        self.enc3 = down_block(128, 256)
        self.enc4 = down_block(256, 512)
        self.enc5 = down_block(512, 512)
        self.enc6 = down_block(512, 512)
        self.enc7 = down_block(512, 512)
        self.enc8 = down_block(512, 512)
        self.dec1 = up_block(512, 512, dropout=True)
        self.dec2 = up_block(1024, 512, dropout=True)
        self.dec3 = up_block(1024, 512, dropout=True)
        self.dec4 = up_block(1024, 512)
        self.dec5 = up_block(1024, 256)
        self.dec6 = up_block(512, 128)
        self.dec7 = up_block(256, 64)
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        d8 = self.dec8(torch.cat([d7, e1], dim=1))
        return d8

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(PatchGANDiscriminator, self).__init__()
        def disc_block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *disc_block(in_channels, 64, normalize=False),
            *disc_block(64, 128),
            *disc_block(128, 256),
            *disc_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

def compute_metrics(pred, gt):
    batch_size = pred.size(0)
    mse_list, ssim_list, psnr_list = [], [], []
    for i in range(batch_size):
        pred_np = pred[i].cpu().detach().numpy().squeeze() * 0.5 + 0.5
        gt_np = gt[i].cpu().detach().numpy().squeeze() * 0.5 + 0.5
        if pred_np.ndim != 2 or gt_np.ndim != 2:
            raise ValueError(f"Invalid image dimensions: pred_np {pred_np.shape}, gt_np {gt_np.shape}")
        mse = np.mean((pred_np - gt_np) ** 2)
        mse_list.append(mse)
        ssim_val = ssim(pred_np, gt_np, data_range=1.0, win_size=7, channel_axis=None)
        ssim_list.append(ssim_val)
        psnr_val = psnr(pred_np, gt_np, data_range=1.0)
        psnr_list.append(psnr_val)
    return np.mean(mse_list), np.mean(ssim_list), np.mean(psnr_list)

def visualize_results(epoch, cond_imgs, gt_imgs, pred_imgs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = 4
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 4 * num_samples))
    #titles = ['Input (with artifacts)', 'Ground Truth', 'Output']
    artifact_name_map = {
        1: "High Frequency Noise",
        2: "Detector Jitter",
        3: "Poisson Noise",
        4: "Motion Blur",
    }

    for i in range(num_samples):
        cond = cond_imgs[i].cpu().squeeze() * 0.5 + 0.5
        gt = gt_imgs[i].cpu().squeeze() * 0.5 + 0.5
        pred = pred_imgs[i].cpu().squeeze() * 0.5 + 0.5

        axes[i][0].imshow(gt, cmap='gray')
        #axes[i][0].set_title(titles[0])
        axes[i][0].set_title("Ground Truth")
        axes[i][0].axis('off')

        axes[i][1].imshow(cond, cmap='gray')
        #axes[i][1].set_title(titles[1])
        axes[i][1].set_title(artifact_name_map.get(i+1, "Unknown"))
        axes[i][1].axis('off')

        axes[i][2].imshow(pred, cmap='gray')
        #axes[i][2].set_title(titles[2])
        axes[i][2].set_title("Reconstruction")
        axes[i][2].axis('off')

        psnr_val = psnr(gt.numpy(), pred.numpy(), data_range=1.0)
        ssim_val = ssim(gt.numpy(), pred.numpy(), data_range=1.0)

        metrics_text = f'PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.3f}'
        axes[i][2].text(1.05, 0.5, metrics_text,
                        transform=axes[i][2].transAxes,
                        verticalalignment='center',
                        fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_subplots(metrics_dict, save_path):
    epochs = metrics_dict["epoch"]
    psnr = metrics_dict["psnr"]
    ssim = metrics_dict["ssim"]
    mse = metrics_dict["mse"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # PSNR
    axs[0].plot(epochs, psnr, marker='o', color='tab:blue')
    axs[0].set_title("PSNR over Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("PSNR (dB)")
    axs[0].grid(True)

    # SSIM
    axs[1].plot(epochs, ssim, marker='s', color='tab:orange')
    axs[1].set_title("SSIM over Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SSIM")
    axs[1].grid(True)

    # MSE
    axs[2].plot(epochs, mse, marker='^', color='tab:red')
    axs[2].set_title("MSE over Epochs")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("MSE")
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
   
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
        print(f"Warning: didn't find labels {missing} in val set.")
    return selected  # list of sample dicts in the same order as labels_to_find but missing ones skipped


def main(train_flag):
    data_dir = '/root_dir/data/diffct/data/train' 
    output_dir = '/root_dir/train/pix2pix_checkpoint/'
    epochs = 550
    batch_size = 8
    lr = 1e-4
    beta1 = 0.5 
    lambda_l1 = 100.0
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    #os.makedirs(os.path.join(output_dir, 'inference'), exist_ok=True)
   
    transform = Compose(
    [
        Lambdad(keys=["gt", "cond"], func=lambda x: x[None, ...]),
        ScaleIntensityRanged(keys=["gt", "cond"], a_min=0.0, a_max=1.0, b_min=-1.0, b_max=1.0, clip=True),
        RandAffined(
            keys=["gt", "cond"],
            spatial_size=[512, 512],
            prob=0.5 
        ),
    ]   
    )
    
    full_dataset = PairedInputDataset(data_dir, transform)                                                
    #test_dataset = PairedInputDataset(test_gt_dir, test_cond_dir, transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    vis_samples = None
    vis_samples = select_val_vis_samples(val_ds, labels_to_find=[1,2,3,4])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = UNetGenerator().to(device)

    if train_flag:
        discriminator = PatchGANDiscriminator().to(device)
        criterion_gan = nn.BCEWithLogitsLoss()
        criterion_l1 = nn.L1Loss()
        optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        metrics = {'train': {'mse': [], 'ssim': [], 'psnr': []}}
        val_metrics = {'epoch': [], 'mse': [], 'ssim': [], 'psnr': []}
        for epoch in range(1, epochs+1):
            generator.train()
            discriminator.train()
            epoch_loss_g = 0.0
            epoch_loss_d = 0.0
            train_mse, train_ssim, train_psnr = [], [], []
            for batch_idx, batch in enumerate(train_loader):
                gt_imgs = batch["gt"].to(device)
                cond_imgs = batch["cond"].to(device)
                optimizer_d.zero_grad()
                real_pair = torch.cat([cond_imgs, gt_imgs], dim=1)
                fake_imgs = generator(cond_imgs)
                fake_pair = torch.cat([cond_imgs, fake_imgs], dim=1)
                real_logits = discriminator(real_pair)
                fake_logits = discriminator(fake_pair.detach())
                label_shape = real_logits.shape
                real_label = torch.ones(label_shape).to(device)
                fake_label = torch.zeros(label_shape).to(device)
                loss_d_real = criterion_gan(real_logits, real_label)
                loss_d_fake = criterion_gan(fake_logits, fake_label)
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                optimizer_d.step()
                optimizer_g.zero_grad()
                fake_logits = discriminator(fake_pair)
                loss_g_gan = criterion_gan(fake_logits, real_label)
                loss_g_l1 = criterion_l1(fake_imgs, gt_imgs) * lambda_l1
                loss_g = loss_g_gan + loss_g_l1
                loss_g.backward()
                optimizer_g.step()

                mse, ssim_val, psnr_val = compute_metrics(fake_imgs, gt_imgs)
                train_mse.append(mse)
                train_ssim.append(ssim_val)
                train_psnr.append(psnr_val)
 
                epoch_loss_d += loss_d.item()
                epoch_loss_g += loss_g.item()

            avg_loss_g = epoch_loss_g / len(train_loader)
            avg_loss_d = epoch_loss_d / len(train_loader)
            metrics['train']['mse'].append(np.mean(train_mse))
            metrics['train']['ssim'].append(np.mean(train_ssim))
            metrics['train']['psnr'].append(np.mean(train_psnr))
            print(f"[Epoch {epoch}/{epochs}] Generator Loss: {avg_loss_g:.4f} | Discriminator Loss: {avg_loss_d:.4f}")
            print(f'Train - MSE: {metrics["train"]["mse"][-1]:.6f}, SSIM: {metrics["train"]["ssim"][-1]:.4f}, PSNR: {metrics["train"]["psnr"][-1]:.4f}')

            if epoch % 25 == 0 and vis_samples:
                current_epoch = epoch
                g_path = os.path.join(output_dir, 'models', f'generator_epoch_{current_epoch}.pth')
                d_path = os.path.join(output_dir, 'models', f'discriminator_epoch_{current_epoch}.pth')
                torch.save(generator.state_dict(), g_path)
                torch.save(discriminator.state_dict(), d_path)
                generator.eval()
                generator.load_state_dict(torch.load(g_path, map_location=device))

                with torch.no_grad():
                    gt_batch = torch.stack([s["gt"] for s in vis_samples]) 
                    cond_batch = torch.stack([s["cond"] for s in vis_samples])       
                    vis_labels = torch.stack([s["label"] for s in vis_samples]).long().to(device)
                    gt_imgs = gt_batch.to(device)
                    cond_imgs = cond_batch.to(device)
                    fake_imgs = generator(cond_imgs)
                    visualize_results(current_epoch, cond_imgs, gt_imgs, fake_imgs, os.path.join(output_dir, 'visualizations'))
            
                epoch_metrics = {'mse': [], 'ssim': [], 'psnr': []}
                with torch.no_grad():
                    for batch in val_loader:
                        cond_img, gt_img = batch["cond"].to(device), batch["gt"].to(device)
                        fake_img = generator(cond_img)
                        mse, ssim_val, psnr_val = compute_metrics(fake_img, gt_img)
                        epoch_metrics['mse'].append(mse)
                        epoch_metrics['ssim'].append(ssim_val)
                        epoch_metrics['psnr'].append(psnr_val)
                    
                avg_metrics = {
                    'epoch': current_epoch,
                    'mse': np.mean(epoch_metrics['mse']),
                    'ssim': np.mean(epoch_metrics['ssim']),
                    'psnr': np.mean(epoch_metrics['psnr']),
                }
            
                val_metrics['epoch'].append(avg_metrics['epoch'])
                val_metrics['mse'].append(avg_metrics['mse'])
                val_metrics['ssim'].append(avg_metrics['ssim'])
                val_metrics['psnr'].append(avg_metrics['psnr'])
                plot_metrics_subplots(val_metrics, os.path.join(output_dir, "val_metrics_plot.png"))
           
        
if __name__ == "__main__":
    main(train_flag=True)
