import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from models.dsgan import DSGenerator, DSDiscriminator
import argparse
from PIL import Image

class GanDataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir, transform=None):
        self.high_res_paths = sorted([os.path.join(high_res_dir, f) for f in os.listdir(high_res_dir) if f.endswith('.png')])
        self.low_res_paths = sorted([os.path.join(low_res_dir, f) for f in os.listdir(low_res_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.high_res_paths)

    def __getitem__(self, idx):
        high_res_image = Image.open(self.high_res_paths[idx]).convert('RGB')
        low_res_image = Image.open(self.low_res_paths[idx]).convert('RGB')

        if self.transform:
            high_res_image = self.transform(high_res_image)
            low_res_image = self.transform(low_res_image)

        return low_res_image, high_res_image

def train_dsgan(high_res_dir, low_res_dir, num_epochs=200, batch_size=64, latent_dim=100, lr=0.0002, beta1=0.5, output_model_dir='models'):
    # Transformations
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Dataset and DataLoader
    dataset = GanDataset(high_res_dir, low_res_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG = DSGenerator().to(device)
    netD = DSDiscriminator().to(device)

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            low_res_images, high_res_images = data
            low_res_images = low_res_images.to(device)
            high_res_images = high_res_images.to(device)
            b_size = low_res_images.size(0)
            
            # Create labels
            real_labels = torch.ones(b_size).to(device)
            fake_labels = torch.zeros(b_size).to(device)
            
            # Train Discriminator
            netD.zero_grad()
            output = netD(high_res_images).view(-1)
            lossD_real = criterion(output, real_labels)
            lossD_real.backward()
            
            fake_images = netG(low_res_images)
            output = netD(fake_images.detach()).view(-1)
            lossD_fake = criterion(output, fake_labels)
            lossD_fake.backward()
            optimizerD.step()
            
            # Train Generator
            netG.zero_grad()
            output = netD(fake_images).view(-1)
            lossG = criterion(output, real_labels)
            lossG.backward()
            optimizerG.step()
            
            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], LossD: {lossD_real.item() + lossD_fake.item()}, LossG: {lossG.item()}')

        # Save models
        os.makedirs(output_model_dir, exist_ok=True)
        torch.save(netG.state_dict(), f'{output_model_dir}/dsgan_generator_{epoch+1}.pth')
        torch.save(netD.state_dict(), f'{output_model_dir}/dsgan_discriminator_{epoch+1}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DS-GAN")
    parser.add_argument('--high_res_dir', type=str, required=True, help='Directory of high-resolution images')
    parser.add_argument('--low_res_dir', type=str, required=True, help='Directory of low-resolution images')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimensionality of the latent vector')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--output_model_dir', type=str, default='models', help='Directory to save trained models')

    args = parser.parse_args()
    train_dsgan(args.high_res_dir, args.low_res_dir, args.num_epochs, args.batch_size, args.latent_dim, args.lr, args.beta1, args.output_model_dir)

