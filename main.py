import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from metrics import InceptionMetrics
from model import Discriminator, Generator, init_weights

DATA = Path("data/landscapes")
OUTPUT = Path("results")
EPOCHS = 350
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)
REAL_LABEL = 0.9
EVAL_EVERY = 10
SEED = 1265


class Paintings(Dataset):
    transform = transforms.Compose([
        transforms.Resize(72), transforms.RandomCrop(64), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def training_paths():
    paths = sorted(DATA.glob("*.jpg"))
    random.Random(SEED).shuffle(paths)
    return paths[:len(paths) - 2 * (len(paths) // 10)]  # hold out 20%


def write_csv(path, header, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def plot_curves(history, scores):
    epochs, d_losses, g_losses = zip(*history)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3))
    for axis, losses, title, color in zip(axes, (d_losses, g_losses),
                                          ("discriminator", "generator"), ("C0", "C1")):
        axis.plot(epochs, losses, color=color)
        axis.set(title=f"{title} loss", xlabel="epoch", ylabel="loss")
    axes[1].set_ylim(top=4)  # the first epochs peak far higher and would flatten the rest
    if scores:
        axes[2].plot([row[0] for row in scores], [row[1] for row in scores], "o-", color="C2")
    axes[2].set(title="FID", xlabel="epoch", ylabel="FID")
    figure.tight_layout()
    figure.savefig(OUTPUT / "training_curves.png", dpi=150)
    plt.close(figure)


@torch.no_grad()
def save_samples(generator, noise, path):
    generator.eval()
    save_image((generator(noise) + 1) / 2, path, nrow=8)
    generator.train()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT.mkdir(exist_ok=True)
    loader = DataLoader(Paintings(training_paths()), batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, num_workers=6)

    generator, discriminator = Generator().to(device), Discriminator().to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    g_optimizer = torch.optim.Adam(generator.parameters(), LEARNING_RATE, betas=BETAS)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), LEARNING_RATE, betas=BETAS)
    criterion = nn.BCEWithLogitsLoss()
    metrics = InceptionMetrics(device, loader)
    fixed_noise = torch.randn(32, generator.latent_dim, device=device)

    history, scores, best_fid = [], [], float("inf")
    for epoch in range(1, EPOCHS + 1):
        d_total = g_total = 0.0
        for real in loader:
            real = real.to(device)
            ones = torch.ones(len(real), 1, device=device)
            fake = generator(torch.randn(len(real), generator.latent_dim, device=device))

            g_optimizer.zero_grad()
            g_loss = criterion(discriminator(fake), ones)
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss = (criterion(discriminator(real), REAL_LABEL * ones)
                      + criterion(discriminator(fake.detach()), 0 * ones)) / 2
            d_loss.backward()
            d_optimizer.step()

            d_total += d_loss.item()
            g_total += g_loss.item()

        history.append((epoch, d_total / len(loader), g_total / len(loader)))
        write_csv(OUTPUT / "history.csv", ["epoch", "d_loss", "g_loss"], history)
        save_samples(generator, fixed_noise, OUTPUT / f"samples_{epoch:03d}.png")
        print(f"epoch {epoch}: D={history[-1][1]:.4f} G={history[-1][2]:.4f}", flush=True)

        if epoch % EVAL_EVERY == 0:
            fid, inception_score = metrics.score(generator)
            scores.append((epoch, fid, inception_score))
            write_csv(OUTPUT / "metrics.csv", ["epoch", "fid", "inception_score"], scores)
            print(f"epoch {epoch}: FID={fid:.2f} IS={inception_score:.3f}", flush=True)
            if fid < best_fid:
                best_fid = fid
                torch.save({"epoch": epoch, "generator": generator.state_dict()},
                           OUTPUT / "best.pt")
        plot_curves(history, scores)

    best =torch.load(OUTPUT / "best.pt", map_location=device)
    generator.load_state_dict(best["generator"])
    save_samples(generator, fixed_noise, OUTPUT / "generated.png")
    print(f"best epoch {best['epoch']}: FID={best_fid:.2f}")


if __name__ == "__main__":
    main()
