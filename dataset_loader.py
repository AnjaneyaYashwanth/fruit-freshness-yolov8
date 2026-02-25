import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class DualStreamDataset(Dataset):
    def __init__(self, base_dir, split="train"):
        self.base_dir = os.path.join(base_dir, split)
        self.classes = ["fresh", "ripe", "overripe"]
        self.samples = []

        for label, cls in enumerate(self.classes):
            rgb_dir = os.path.join(self.base_dir, cls, "rgb")

            if not os.path.exists(rgb_dir):
                continue

            images = os.listdir(rgb_dir)

            for img_name in images:
                self.samples.append((cls, img_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls, img_name, label = self.samples[idx]

        rgb_path = os.path.join(self.base_dir, cls, "rgb", img_name)
        edge_path = os.path.join(self.base_dir, cls, "edge", img_name)
        lbp_path = os.path.join(self.base_dir, cls, "lbp", img_name)

        rgb = cv2.imread(rgb_path)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        lbp = cv2.imread(lbp_path, cv2.IMREAD_GRAYSCALE)

        # Normalize
        rgb = rgb / 255.0
        edge = edge / 255.0
        lbp = lbp / 255.0

        # Convert to tensor
        rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)
        edge = torch.tensor(edge, dtype=torch.float32).unsqueeze(0)
        lbp = torch.tensor(lbp, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return rgb, edge, lbp, label