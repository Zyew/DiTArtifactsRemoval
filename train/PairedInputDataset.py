import os, torch
import numpy as np
from monai.data import Dataset

class PairedInputDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.slice_infos = []
        self.artifact_label_map = {
            "hf": 1,
            "jitter": 2,
            "poisson": 3,
            "motion": 4,
        }
        #label_counter = 0  

        for organ in sorted(os.listdir(root_dir)):
            organ_path = os.path.join(root_dir, organ)

            gt_dir = os.path.join(organ_path, "gt")
            if not os.path.isdir(gt_dir):
                continue
            
            # List all condition directories except "gt"
            cond_types = [
                d for d in os.listdir(organ_path)
                if d != "gt" and os.path.isdir(os.path.join(organ_path, d))
            ]

            for cond_type in cond_types:
                if cond_type not in self.artifact_label_map:
                    continue
                cond_dir = os.path.join(organ_path, cond_type)
                filenames = set(os.listdir(gt_dir)) & set(os.listdir(cond_dir))
                for fname in sorted(filenames):
                    if not fname.endswith(".npy"):
                        continue
                    gt_path = os.path.join(gt_dir, fname)
                    cond_path = os.path.join(cond_dir, fname)

                    gt_volume = np.load(gt_path)
                    cond_volume = np.load(cond_path)
                    if gt_volume.shape != cond_volume.shape:
                        print(f"Shape mismatch: {fname} in {organ}/{cond_type}")
                        continue

                    num_slices = gt_volume.shape[0]
                    for slice_idx in range(num_slices):
                        self.slice_infos.append(
                            (gt_path, cond_path, slice_idx, cond_type)
                        )

    def __len__(self):
        return len(self.slice_infos)

    def __getitem__(self, idx):
        gt_path, cond_path, slice_idx, artifact_type = self.slice_infos[idx]

        gt_volume = np.load(gt_path)
        cond_volume = np.load(cond_path)

        gt_slice = gt_volume[slice_idx]
        cond_slice = cond_volume[slice_idx]
        label = self.artifact_label_map[artifact_type]

        sample = {
            "gt": gt_slice,
            "cond": cond_slice,
            "label": torch.tensor(label, dtype=torch.long)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample