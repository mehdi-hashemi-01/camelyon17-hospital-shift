import csv
from datetime import timedelta
from time import perf_counter

started = perf_counter()
print("[1/4] Loading PyTorch and torchvision...", flush=True)

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18

import settings
from settings import BATCH_SIZE, MODEL, ROOT, image_transform

LOG_INTERVAL_SECONDS = 15
device = "cuda" if torch.cuda.is_available() else "cpu"
device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
print(f"Device: {device_name}", flush=True)

print(f"[2/4] Loading checkpoint from {MODEL}...", flush=True)
model = resnet18(weights=None, num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL, map_location=device, weights_only=True))
model.eval()

print("[3/4] Reading Camelyon17 metadata...", flush=True)
if hasattr(settings, "load_camelyon17"):
    data = settings.load_camelyon17()
else:
    from wilds import get_dataset

    data = get_dataset(dataset="camelyon17", root_dir=str(settings.DATA), download=False)

print("[4/4] Evaluating validation splits...", flush=True)
summary = []
with torch.inference_mode():
    for split in ("id_val", "val"):
        subset = data.get_subset(split, transform=image_transform())
        n = len(subset)
        if n == 0:
            raise ValueError(f"No validation patches in split {split!r}.")
        loader = DataLoader(subset, batch_size=BATCH_SIZE)
        print(
            f"[{split}] {n:,} patches | {len(loader):,} batches | "
            f"batch size {BATCH_SIZE} | reading first batch...",
            flush=True,
        )
        split_start = last_log = perf_counter()
        processed = correct = 0
        confidence_sum = brier_sum = 0.0
        for batch, (images, labels, _) in enumerate(loader, start=1):
            probs = model(images.to(device)).softmax(dim=1).cpu()
            confidence, predictions = probs.max(dim=1)
            correct += (predictions == labels).sum().item()
            confidence_sum += confidence.sum().item()
            brier_sum += ((probs[:, 1] - labels) ** 2).sum().item()
            processed += len(labels)

            now = perf_counter()
            if batch == 1 or batch == len(loader) or now - last_log >= LOG_INTERVAL_SECONDS:
                elapsed = now - split_start
                speed = processed / elapsed
                remaining = (n - processed) / speed
                print(
                    f"[{split}] Batch {batch:,}/{len(loader):,} | {processed:,}/{n:,} "
                    f"({processed / n:.1%}) | accuracy {correct / processed:.4f} | "
                    f"{speed:.1f} patches/s | elapsed {timedelta(seconds=int(elapsed))} | "
                    f"ETA {timedelta(seconds=round(remaining))}",
                    flush=True,
                )
                last_log = now

        summary.append([split, n, correct / n, confidence_sum / n, brier_sum / n])
        print(
            f"[{split}] accuracy={correct / n:.4f} | mean_confidence={confidence_sum / n:.4f} | "
            f"binary_brier={brier_sum / n:.4f}",
            flush=True,
        )

output = ROOT / "results/validation_metrics.csv"
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["split", "patches", "accuracy", "mean_confidence", "binary_brier"])
    writer.writerows(summary)
print(f"Saved {output} | total time {timedelta(seconds=int(perf_counter() - started))}", flush=True)
