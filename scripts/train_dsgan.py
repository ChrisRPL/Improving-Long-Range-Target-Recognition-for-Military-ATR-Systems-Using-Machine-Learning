# scripts/train_dsgan.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.dsgan import DSGenerator, DSDiscriminator

# Hyperparameters
batch_size = 64
image_size = 64
latent_dim = 100
num_epochs = 200
lr = 0.0002
beta1 = 0.5

# Transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset and DataLoader
dataset = datasets.ImageFolder(root='data/gan_dataset', transform=transform)
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
        real_images, _ = data
        real_images = real_images.to(device)
        b_size = real_images.size(0)
        
        # Create labels
        real_labels = torch.ones(b_size).to(device)
        fake_labels = torch.zeros(b_size).to(device)
        
        # Train Discriminator
        netD.zero_grad()
        output = netD(real_images).view(-1)
        lossD_real = criterion(output, real_labels)
        lossD_real.backward()
        
        noise = torch.randn(b_size, latent_dim, 1, 1).to(device)
        fake_images = netG(noise)
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
    torch.save(netG.state_dict(), f'models/dsgan_generator_{epoch+1}.pth')
    torch.save(netD.state_dict(), f'models/dsgan_discriminator_{epoch+1}.pth')

