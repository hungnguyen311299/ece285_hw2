import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 128
        widths = [32, 64, 128, 256, 512]

        def block(cin, cout, up=False):
            conv = (nn.ConvTranspose2d(cin, cout, 3, 2, 1, output_padding=1)
                    if up else nn.Conv2d(cin, cout, 3, 2, 1))
            return nn.Sequential(conv, nn.BatchNorm2d(cout), nn.LeakyReLU())

        self.encoder = nn.Sequential(*[
            block(cin, cout) for cin, cout in zip([1] + widths[:-1], widths)
        ])
        self.fc_mu = nn.Linear(2048, self.latent_dim)
        self.fc_var = nn.Linear(2048, self.latent_dim)
        self.decoder_input = nn.Linear(self.latent_dim, 2048)
        reverse = widths[::-1]
        self.decoder = nn.Sequential(*[
            block(cin, cout, up=True) for cin, cout in zip(reverse[:-1], reverse[1:])
        ])
        self.final_layer = nn.Sequential(
            *block(32, 32, up=True), nn.Conv2d(32, 1, 3, padding=1), nn.Tanh()
        )

    def encode(self, images):
        features = self.encoder(images).flatten(1)
        return self.fc_mu(features), self.fc_var(features)

    def decode(self, z):
        features = self.decoder_input(z).reshape(-1, 512, 2, 2)
        return self.final_layer(self.decoder(features))

    def forward(self, images):
        mean, log_variance = self.encode(images)
        z = mean + torch.exp(log_variance / 2) * torch.randn_like(mean)
        return self.decode(z), mean, log_variance


def objective(images, reconstruction, mean, log_variance):
    mse = (images - reconstruction).square().mean()
    kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(1).mean()
    return mse + 0.00025 * kl, mse, kl
