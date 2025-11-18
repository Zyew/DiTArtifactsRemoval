import os
import numpy as np
from monai.data import Dataset

class ArtifactsRemovalDataset(Dataset):
    def __init__(self, gt_root, cond_root, transform=None):
        self.gt_root = gt_root
        self.cond_root = cond_root
        self.transform = transform
        filenames = sorted([
            os.path.splitext(f)[0] 
            for f in os.listdir(gt_root)
            if f.endswith('.npy') and os.path.exists(os.path.join(cond_root, f))
        ])

        self.slice_infos = [] 

        for name in filenames:
            gt_path = os.path.join(gt_root, f"{name}.npy")
            cond_path = os.path.join(cond_root, f"{name}.npy")

            if not (os.path.exists(gt_path) and os.path.exists(cond_path)):
                continue

            gt_volume = np.load(gt_path)
            cond_volume = np.load(cond_path)
            assert gt_volume.shape == cond_volume.shape, f"Shape mismatch: {gt_path} vs {cond_path}"
            num_slices = gt_volume.shape[0]
            for slice_idx in range(num_slices):
                self.slice_infos.append((gt_path, cond_path, slice_idx))
  
        
    def __len__(self):
        return len(self.slice_infos)

    def __getitem__(self, idx):
        gt_path, cond_path, slice_idx = self.slice_infos[idx]
        gt_volume = np.load(gt_path).astype(np.float32)
        cond_volume = np.load(cond_path).astype(np.float32)
        gt_slice = gt_volume[slice_idx]
        cond_slice = cond_volume[slice_idx]

        sample = {
            "gt": (gt_slice),
            "cond": (cond_slice),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
        