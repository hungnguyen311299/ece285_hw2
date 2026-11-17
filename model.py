from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.project_noise = nn.Linear(latent_dim, 512 * 4 * 4)

        def up(cin, cout):
            return [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]

        self.synthesis = nn.Sequential(
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            *up(512, 256), *up(256, 128), *up(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z):
        return self.synthesis(self.project_noise(z).view(-1, 512, 4, 4))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def down(cin, cout):
            return [nn.Conv2d(cin, cout, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True)]

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            *down(64, 128), *down(128, 256), *down(256, 512),
        )
        self.classify = nn.Linear(512 * 4 * 4, 1)

    def forward(self, images):
        return self.classify(self.features(images).flatten(1))


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
