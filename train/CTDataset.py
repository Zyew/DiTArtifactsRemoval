import os
import numpy as np
from monai.data import Dataset

class CTDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.slice_infos = [] 

        for dirpath, _, filenames in os.walk(root):
            for fname in sorted(filenames):
                if not fname.endswith('.npy'):
                    continue

                file_path = os.path.join(dirpath, fname)
                try:
                    volume = np.load(file_path)
                    if volume.ndim == 4:
                        volume = volume[0]  

                    num_slices = volume.shape[0]

                  
                    rel_path = os.path.relpath(file_path, root)
                    organ = rel_path.split(os.sep)[0]  

                    for slice_idx in range(num_slices):
                        self.slice_infos.append((file_path, slice_idx, organ))
                except Exception as e:
                    print(f"[Warning] Skipping {file_path}: {e}")

    def __len__(self):
        return len(self.slice_infos)

    def __getitem__(self, idx):
        file_path, slice_idx, organ = self.slice_infos[idx]
        volume = np.load(file_path)

        if volume.ndim == 4:
            volume = volume[0]

        slice_img = volume[slice_idx]

        sample = {
            "image": slice_img,
            "organ": organ
        }

        if self.transform:
            sample = self.transform(sample)

        return sample