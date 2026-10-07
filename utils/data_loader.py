import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

# === AUGMENTATIONS ===
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# TTA: 5-crop + flip (unchanged)
TTA_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops
    ])),
])

class ReplayBuffer:
    """Simple reservoir sampling for ER buffer."""
    def __init__(self, max_size):
        self.buffer = []  # List of (img_path, label)
        self.max_size = max_size

    def add_samples(self, samples):
        """Add new samples; reservoir sample if full."""
        for path, label in samples:
            if len(self.buffer) < self.max_size:
                self.buffer.append((path, label))
            else:
                # Reservoir: replace random old sample
                idx = random.randint(0, len(self.buffer) - 1)
                self.buffer[idx] = (path, label)
        random.shuffle(self.buffer)  # Keep diverse

    def sample(self, n):
        """Sample n items (with replacement if needed)."""
        if len(self.buffer) == 0:
            return []
        return random.choices(self.buffer, k=n)

class SkinCancerDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, validation=False):
        self.root_dir = root_dir
        self.split = split
        self.classes = ["benign", "malignant"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data = []  # List of (img_path, label)

        if transform is None:
            if split == "train":
                self.transform = TRAIN_TRANSFORM
            else:
                self.transform = VAL_TEST_TRANSFORM
        else:
            self.transform = transform

        # Load initial data
        self._load_initial_data(validation)

        print(f"[{split.upper()}] Loaded {len(self.data)} samples from {root_dir}/{split}")

    def _load_initial_data(self, validation):
        """Load from subdirs (benign/malignant)."""
        split_path = os.path.join(self.root_dir, self.split)
        for cls in self.classes:
            cls_path = os.path.join(split_path, cls)
            if not os.path.exists(cls_path):
                continue  # Skip if missing (for dynamic-only)
            for img_file in os.listdir(cls_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append((os.path.join(cls_path, img_file), self.class_to_idx[cls]))

        if validation and self.split == "train":
            random.seed(42)
            random.shuffle(self.data)
            val_size = int(0.2 * len(self.data))
            self.data = self.data[val_size:] if not validation else self.data[:val_size]

    def reload(self, new_samples=None, replay_buffer=None):
        """Append new samples dynamically. new_samples: list of (img_path, label)."""
        if new_samples:
            # Validate new samples (optional: check paths exist, labels in 0-1)
            valid_new = [(path, label) for path, label in new_samples if os.path.exists(path) and label in self.class_to_idx.values()]
            self.data.extend(valid_new)
            print(f"[RELOAD] Added {len(valid_new)} valid new samples (total: {len(self.data)})")
            # Add to replay buffer if provided
            if replay_buffer:
                replay_buffer.add_samples(valid_new)
        # Shuffle for better training
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)  # Ensure long for CrossEntropy

class ReplaySkinCancerDataset(SkinCancerDataset):
    """Wrapper for continual learning: mixes current + replay buffer."""
    def __init__(self, root_dir, split="train", transform=None, replay_buffer=None, replay_ratio=0.5, validation=False):
        super().__init__(root_dir, split, transform, validation)
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio
        self.current_len = len(self.data)  # Track for mixing

    def __len__(self):
        # Effective len: current + proportional replay
        replay_len = int(self.replay_ratio * self.current_len) if self.replay_buffer else 0
        return self.current_len + replay_len

    def __getitem__(self, idx):
        # Mix: first half current, second replay
        if idx < self.current_len:
            return super().__getitem__(idx % len(self.data))  # FIXED: Use self.data, not super().data
        else:
            # Sample from buffer
            replay_idx = idx - self.current_len
            replay_path, replay_label = self.replay_buffer.sample(1)[0]
            replay_img = Image.open(replay_path).convert("RGB")
            if self.transform:
                replay_img = self.transform(replay_img)
            return replay_img, torch.tensor(replay_label, dtype=torch.long)