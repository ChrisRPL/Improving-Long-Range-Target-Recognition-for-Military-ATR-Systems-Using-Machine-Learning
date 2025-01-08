import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
from models.dsgan import DSGenerator, DSDiscriminator

class DSGANDataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir, image_size=32):
        self.high_res_dir = Path(high_res_dir)
        self.low_res_dir = Path(low_res_dir)
        self.image_pairs = self._get_image_pairs()
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _get_image_pairs(self):
        high_res_files = set(f.name for f in self.high_res_dir.glob('*.png'))
        low_res_files = set(f.name for f in self.low_res_dir.glob('*.png'))
        return sorted(list(high_res_files & low_res_files))  # Get common files

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image_name = self.image_pairs[idx]
        high_res = Image.open(self.high_res_dir / image_name).convert('RGB')
        low_res = Image.open(self.low_res_dir / image_name).convert('RGB')
        
        return self.transform(low_res), self.transform(high_res)

class DSGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_models()
        self.setup_data()
        self.setup_training()

    def setup_models(self):
        self.netG = DSGenerator().to(self.device)
        self.netD = DSDiscriminator().to(self.device)

    def setup_data(self):
        dataset = DSGANDataset(self.config.high_res_dir, self.config.low_res_dir)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

    def setup_training(self):
        self.criterion = nn.BCELoss()
        self.optimG = optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))
        self.optimD = optim.Adam(self.netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))

    def train_epoch(self, epoch):
        for i, (low_res, high_res) in enumerate(self.dataloader):
            batch_size = low_res.size(0)
            real_label = torch.ones(batch_size, device=self.device)
            fake_label = torch.zeros(batch_size, device=self.device)

            # Update Discriminator
            self.netD.zero_grad()
            real_output = self.netD(high_res.to(self.device))
            errD_real = self.criterion(real_output, real_label)
            errD_real.backward()

            fake = self.netG(low_res.to(self.device))
            fake_output = self.netD(fake.detach())
            errD_fake = self.criterion(fake_output, fake_label)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            self.optimD.step()

            # Update Generator
            self.netG.zero_grad()
            output = self.netD(fake)
            errG = self.criterion(output, real_label)
            errG.backward()
            self.optimG.step()

            if i % 100 == 0:
                print(f'[{epoch}/{self.config.num_epochs}][{i}/{len(self.dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

    def save_checkpoint(self, epoch):
        checkpoint_dir = Path(self.config.output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.netG.state_dict(),
            'discriminator_state_dict': self.netD.state_dict(),
            'optimG_state_dict': self.optimG.state_dict(),
            'optimD_state_dict': self.optimD.state_dict(),
        }, checkpoint_dir / f'dsgan_checkpoint_epoch_{epoch}.pt')

    def train(self):
        for epoch in range(1, self.config.num_epochs + 1):
            self.train_epoch(epoch)
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch)

def main():
    parser = argparse.ArgumentParser(description="Train DS-GAN")
    parser.add_argument('--high_res_dir', type=str, required=True)
    parser.add_argument('--low_res_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--save_interval', type=int, default=10)
    
    config = parser.parse_args()
    trainer = DSGANTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
