import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3


class InceptionMetrics:
    """FID (2048-d pool features) and Inception Score against a fixed set of real images."""

    def __init__(self, device, real_loader):
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.classifier = inception.fc.to(device).eval()
        inception.fc = torch.nn.Identity()
        self.inception = inception.to(device).eval()
        self.device = device
        real = torch.cat([self.features(images) for images in real_loader]).cpu().numpy()
        self.count = len(real)
        self.real_mean, self.real_cov = real.mean(0), np.cov(real, rowvar=False)
        values, vectors = np.linalg.eigh(self.real_cov)
        self.real_cov_sqrt = (vectors * np.sqrt(values.clip(min=0))) @ vectors.T

    @torch.no_grad()
    def features(self, images):
        images = ((images.to(self.device) + 1) / 2).clamp(0, 1)
        return self.inception(F.interpolate(images, size=(299, 299), mode="bilinear"))

    @torch.no_grad()
    def score(self, generator, seed=0):
        generator.eval()
        noise = torch.randn(self.count, generator.latent_dim,
                            generator=torch.Generator().manual_seed(seed))
        fake = torch.cat([self.features(generator(z.to(self.device))) for z in noise.split(100)])
        generator.train()

        log_probs = F.log_softmax(self.classifier(fake), dim=1).double()
        marginal = log_probs.exp().mean(0).log()
        inception_score = (log_probs.exp() * (log_probs - marginal)).sum(1).mean().exp().item()

        fake = fake.double().cpu().numpy()
        mean, cov = fake.mean(0), np.cov(fake, rowvar=False)
        # tr sqrt(A B) = tr sqrt(sqrt(A) B sqrt(A)), which stays stable for singular covariances.
        middle = self.real_cov_sqrt @ cov @ self.real_cov_sqrt
        trace_sqrt = np.sqrt(np.linalg.eigvalsh((middle + middle.T) / 2).clip(min=0)).sum()
        fid = ((self.real_mean - mean) ** 2).sum() + np.trace(self.real_cov + cov) - 2 * trace_sqrt
        return float(fid), inception_score
