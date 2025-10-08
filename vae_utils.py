import os
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
from collections import defaultdict

def get_timepoint_dirs(root_dir):
    timepoint_dict = defaultdict(list)
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path) and len(name) >= 2 and name[:2].isdigit():
            timepoint = name[:2]
            timepoint_dict[timepoint].append(path)
    return timepoint_dict

def read_dvf(dvf_path):
    img = sitk.ReadImage(dvf_path)
    array = sitk.GetArrayFromImage(img).astype(np.float32)
    return array, img

def save_dvf(array, reference_img, save_path):
    dvf_img = sitk.GetImageFromArray(array.astype(np.float64), isVector=True)
    dvf_img.CopyInformation(reference_img)
    sitk.WriteImage(dvf_img, save_path)

class DVFDataset(Dataset):
    def __init__(self, folders):
        self.paths = [os.path.join(f, "DVF.nii") for f in folders if os.path.exists(os.path.join(f, "DVF.nii"))]
        self.images = []
        for path in self.paths:
            arr, _ = read_dvf(path)
            self.images.append(arr)
        self.images = np.stack(self.images)
        self.images = torch.tensor(self.images, dtype=torch.float32)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]
