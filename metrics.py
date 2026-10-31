import math
import numpy as np
import torch
from PIL import Image
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from torchvision.transforms import functional as TF

class StableFID(FID):
    def compute(self):
        self.used_fallback = False
        try:
            score = super().compute()
            if math.isfinite(score) and score >= -1e-6:
                return max(0.0, score)
        except ValueError as error:
            if "Imaginary component" not in str(error):
                raise
        self.used_fallback = True
        count = self._num_examples
        a = self._get_covariance(self._train_sigma, self._train_total).cpu().numpy()
        b = self._get_covariance(self._test_sigma, self._test_total).cpu().numpy()
        a, b = (a + a.T) / 2, (b + b.T) / 2
        values, vectors = np.linalg.eigh(a)
        sqrt_a = (vectors * np.sqrt(values.clip(min=0))) @ vectors.T
        middle = sqrt_a @ b @ sqrt_a
        trace_sqrt = np.sqrt(np.linalg.eigvalsh((middle + middle.T) / 2).clip(min=0)).sum()
        delta = ((self._train_total - self._test_total) / count).cpu().numpy()
        score = float(delta @ delta + np.trace(a) + np.trace(b) - 2 * trace_sqrt)
        if not math.isfinite(score):
            raise RuntimeError("FID covariance calculation produced a non-finite result")
        return max(0.0, score)


def inception_images(images):
    if images.ndim != 4 or images.shape[1] not in (1, 3):
        raise ValueError("Expected [N, 1 or 3, H, W] images")
    if not torch.isfinite(images).all():
        raise ValueError("Images contain non-finite values")
    images = ((images.detach().cpu().float() + 1) / 2).clamp(0, 1)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    bilinear = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    return torch.stack([
        TF.to_tensor(TF.to_pil_image(image).resize((299, 299), bilinear))
        for image in images
    ])


class GenerativeEvaluator:
    def __init__(self, device="cpu", feature_extractor=None, num_features=1000):
        self.device = torch.device(device)
        cuda_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            self.classifier = (feature_extractor if feature_extractor is not None
                               else InceptionModel(return_features=False, device=self.device))
        self.classifier.to(self.device).eval()
        self.num_features = num_features

    def evaluate(self, model, loader, max_samples=None, seed=1265):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise ValueError("Generation metrics currently require a single process/device")
        count = len(loader.dataset)
        if max_samples is not None:
            if max_samples < 2:
                raise ValueError("max_samples must be at least 2")
            count = min(count, max_samples)
        if count < 2:
            raise ValueError("FID requires at least two real and generated images")
        generator = torch.Generator().manual_seed(seed)
        seen = 0

        def batches():
            remaining = count
            for images, _ in loader:
                images = images[:remaining]
                yield images
                remaining -= images.shape[0]
                if remaining == 0:
                    break

        @torch.no_grad()
        def step(engine, images):
            nonlocal seen
            z = torch.randn(images.shape[0], model.latent_dim, generator=generator)
            fake = model.decode(z.to(self.device))
            fake_probs = self.classifier(inception_images(fake).to(self.device))
            real_probs = self.classifier(inception_images(images).to(self.device))
            seen += images.shape[0]
            return fake_probs, real_probs

        engine = Engine(step)
        fid = StableFID(num_features=self.num_features, device=self.device)
        fid.attach(engine, "fid")
        InceptionScore(num_features=self.num_features, device=self.device,
                       output_transform=lambda output: output[0]).attach(engine, "inception_score")
        was_training = model.training
        cuda_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            model.eval()
            with torch.random.fork_rng(devices=cuda_devices):
                result = dict(engine.run(batches()).metrics)
        finally:
            model.train(was_training)
        if not all(math.isfinite(float(value)) for value in result.values()):
            raise RuntimeError("Non-finite generation metric; check images and sample count")
        result.update(num_real=seen, num_generated=seen, seed=seed,
                      batch_size=loader.batch_size,
                      fid_features=self.num_features,
                      fid_algorithm=("symmetric_psd_eigen_fallback" if fid.used_fallback else "ignite_scipy_sqrtm"),
                      fid_convention="ignite_inception_probabilities",
                      resize="PIL bilinear 299x299", input_range="[-1,1] to [0,1]")
        return result
