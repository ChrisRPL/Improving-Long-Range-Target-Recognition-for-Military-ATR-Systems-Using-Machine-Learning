import torch
import torch.nn as nn

class DSGenerator(nn.Module):
    def __init__(self, input_channels=3, base_filters=64):
        super(DSGenerator, self).__init__()
        
        # Encoder (downsampling)
        self.encoder = nn.Sequential(
            # input is (3) x 32 x 32
            nn.Conv2d(input_channels, base_filters, 3, 1, 1),  # -> (64) x 32 x 32
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_filters, base_filters * 2, 3, 2, 1),  # -> (128) x 16 x 16
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, 2, 1),  # -> (256) x 8 x 8
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 4, 2, 1),  # -> (128) x 16 x 16
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(base_filters * 2, base_filters, 4, 2, 1),  # -> (64) x 32 x 32
            nn.BatchNorm2d(base_filters),
            nn.ReLU(True),
            
            nn.Conv2d(base_filters, input_channels, 3, 1, 1),  # -> (3) x 32 x 32
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DSDiscriminator(nn.Module):
    def __init__(self, input_channels=3, base_filters=64):
        super(DSDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # input is (3) x 32 x 32
            nn.Conv2d(input_channels, base_filters, 3, 2, 1),  # -> (64) x 16 x 16
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_filters, base_filters * 2, 3, 2, 1),  # -> (128) x 8 x 8
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, 2, 1),  # -> (256) x 4 x 4
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_filters * 4, 1, 4, 1, 0),  # -> (1) x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)
