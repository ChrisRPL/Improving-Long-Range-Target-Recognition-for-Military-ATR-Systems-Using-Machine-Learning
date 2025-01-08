import torch
import torch.nn as nn

class DSGenerator(nn.Module):
    def __init__(self, input_channels=3, base_filters=64):
        super(DSGenerator, self).__init__()
        
        # Encoder (downsampling)
        self.encoder = nn.Sequential(
            self._encoder_block(input_channels, base_filters, batch_norm=False),  # 64x64 -> 32x32
            self._encoder_block(base_filters, base_filters * 2),                  # 32x32 -> 16x16
            self._encoder_block(base_filters * 2, base_filters * 4),             # 16x16 -> 8x8
            self._encoder_block(base_filters * 4, base_filters * 8),             # 8x8 -> 4x4
        )
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            self._decoder_block(base_filters * 8, base_filters * 4),             # 4x4 -> 8x8
            self._decoder_block(base_filters * 4, base_filters * 2),             # 8x8 -> 16x16
            self._decoder_block(base_filters * 2, base_filters),                 # 16x16 -> 32x32
            self._decoder_block(base_filters, input_channels, batch_norm=False, last_layer=True)  # 32x32 -> 64x64
        )

    def _encoder_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, last_layer=False):
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True) if not last_layer else nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DSDiscriminator(nn.Module):
    def __init__(self, input_channels=3, base_filters=64):
        super(DSDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            self._conv_block(input_channels, base_filters, batch_norm=False),    # 64x64 -> 32x32
            self._conv_block(base_filters, base_filters * 2),                    # 32x32 -> 16x16
            self._conv_block(base_filters * 2, base_filters * 4),               # 16x16 -> 8x8
            self._conv_block(base_filters * 4, base_filters * 8),               # 8x8 -> 4x4
            nn.Conv2d(base_filters * 8, 1, kernel_size=4, stride=1, padding=0), # 4x4 -> 1x1
            nn.Sigmoid()
        )

    def _conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)
