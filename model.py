import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=2, padding=1),
            nn.Conv1d(16, 32, 3, stride=2, padding=1),
            nn.Conv1d(32, 64, 3)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 3),
            nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose1d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
