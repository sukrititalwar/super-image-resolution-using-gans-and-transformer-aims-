import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SwinIRForImageSuperResolution, SwinIRImageProcessor

# Paths
HR_DIR = './data/train/HR'
LR_DIR = './data/train/LR'
BATCH_SIZE = 4
EPOCHS = 20
LR_SIZE = (24, 24)
HR_SIZE = (96, 96)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = 'swinir_finetuned.pth'

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, hr_size, lr_size):
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize(lr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        assert len(self.hr_files) == len(self.lr_files), "HR and LR image counts do not match."

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert('RGB')
        lr = Image.open(self.lr_files[idx]).convert('RGB')
        hr = self.hr_transform(hr)
        lr = self.lr_transform(lr)
        return lr, hr

dataset = SRDataset(HR_DIR, LR_DIR, HR_SIZE, LR_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# --- Updated model import and loading ---
model = SwinIRForImageSuperResolution.from_pretrained("caidas/swinir-classical-x4")
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for lr, hr in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)
        outputs = model(pixel_values=lr, labels=hr)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * lr.size(0)
    avg_loss = epoch_loss / len(dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Avg MSE Loss: {avg_loss:.6f}")

    # Save checkpoint
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'swinir_epoch_{epoch+1}.pth')

torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")