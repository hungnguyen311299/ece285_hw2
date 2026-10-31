import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import VAE, objective


def data_root(path):
    if path is None:
        import kagglehub
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    root = Path(path).expanduser().resolve()
    for candidate in (root, root / "chest_xray", root / "chest_xray/chest_xray"):
        if all((candidate / split).is_dir() for split in ("train", "val", "test")):
            return candidate
    raise FileNotFoundError(f"No train/val/test directories under {root}")


def loader(root, split, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(), transforms.Resize((64, 64)),
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ])
    images = datasets.ImageFolder(root / split, transform=transform)
    target = images.class_to_idx["NORMAL"]
    indices = [i for i, label in enumerate(images.targets) if label == target]
    return DataLoader(Subset(images, indices), batch_size=batch_size, shuffle=split == "train")


def epoch(model, batches, device, optimizer=None):
    model.train(optimizer is not None)
    sums = torch.zeros(3, device=device)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, _ in batches:
            images = images.to(device)
            reconstruction, mean, log_variance = model(images)
            terms = objective(images, reconstruction, mean, log_variance)
            if optimizer is not None:
                optimizer.zero_grad()
                terms[0].backward()
                optimizer.step()
            sums += torch.stack([term.detach() for term in terms]) * len(images)
            count += len(images)
    return (sums / count).cpu().tolist()


def load(model, checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict({k.removeprefix("model."): v for k, v in saved["state_dict"].items()})


@torch.no_grad()
def preview(model, output, device, filename="generated.png"):
    model.eval()
    z = torch.randn(32, model.latent_dim, generator=torch.Generator().manual_seed(1265))
    images = ((model.decode(z.to(device)) + 1) / 2).clamp(0, 1)
    save_image(images, output / filename, nrow=8)


def train(model, root, output, device, epochs, batch_size):
    training = loader(root, "train", batch_size)
    validation = loader(root, "val", batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, best = [], float("inf")
    fields = ["epoch"] + [f"{split}_{metric}" for split in ("train", "val") for metric in ("loss", "mse", "kl")]
    for index in range(1, epochs + 1):
        values = epoch(model, training, device, optimizer) + epoch(model, validation, device)
        history.append(dict(zip(fields, [index] + values)))
        with (output / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        if values[3] < best:
            best = values[3]
            torch.save({"state_dict": model.state_dict(), "epoch": index, "val_loss": best}, output / "best.pt")
        preview(model, output, device, f"samples_{index:03d}.png")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for ax, metric in zip(axes, ("loss", "mse", "kl")):
            for split in ("train", "val"):
                ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=split)
            ax.set(xlabel="Epoch", title=metric)
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / "loss_curves.png", dpi=150)
        plt.close(fig)
        print(f"Epoch {index}/{epochs}: train={values[0]:.5f}, val={values[3]:.5f}", flush=True)
    load(model, output / "best.pt", device)
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(validation))
        images = images[:8].to(device)
        mean, _ = model.encode(images)
        pairs = torch.stack((images, model.decode(mean)), dim=1).flatten(0, 1)
        save_image(((pairs + 1) / 2).clamp(0, 1), output / "reconstructions.png", nrow=8)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a VAE on normal chest X-ray images."
    )
    parser.add_argument("--data-path", help="KaggleHub path; omitted means download automatically")
    parser.add_argument("--checkpoint", help="Skip training and load best.pt or an existing VanillaVAE .ckpt")
    parser.add_argument("--evaluate", action="store_true", help="Compute test FID/IS after training or loading")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    torch.manual_seed(1265)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    model = VAE().to(device)
    root = data_root(args.data_path) if not args.checkpoint or args.evaluate else None
    if args.checkpoint:
        load(model, args.checkpoint, device)
    else:
        train(model, root, args.output, device, args.epochs, args.batch_size)
    preview(model, args.output, device)
    if args.evaluate:
        from metrics import GenerativeEvaluator
        scores = GenerativeEvaluator(device).evaluate(model, loader(root, "test", args.batch_size))
        scores.update(split="test", class_name="NORMAL")
        (args.output / "metrics.json").write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n")
        print(json.dumps(scores, indent=2))
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
