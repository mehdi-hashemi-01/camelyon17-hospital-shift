from datetime import timedelta
from time import perf_counter

started = perf_counter()
print("[1/4] Loading PyTorch and torchvision...", flush=True)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from settings import BATCH_SIZE, load_camelyon17, MODEL, SEED, image_transform

LOG_INTERVAL_SECONDS = 15

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
print(f"Device: {device_name}", flush=True)

print("[2/4] Reading Camelyon17 metadata and preparing training batches...", flush=True)
data = load_camelyon17()
subset = data.get_subset("train", transform=image_transform())
train = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
print(f"{len(subset):,} patches | {len(train):,} batches | batch size {BATCH_SIZE}", flush=True)

print("[3/4] Loading ResNet-18 ImageNet weights and setting up the linear head...", flush=True)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.requires_grad_(False)
model.fc = nn.Linear(model.fc.in_features, 2)
# Keep pretrained batch-normalization statistics fixed.
model.to(device).eval()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print("[4/4] Epoch 1/1: reading the first batch of images...", flush=True)
training_start = last_log = perf_counter()
processed = 0
for batch, (images, labels, _) in enumerate(train, start=1):
    if batch == 1:
        print("First batch loaded. Training the linear head...", flush=True)
    optimizer.zero_grad()
    logits = model(images.to(device))
    loss = loss_fn(logits, labels.to(device))
    loss.backward()
    optimizer.step()
    processed += len(labels)

    if batch == 1 or batch == len(train) or perf_counter() - last_log >= LOG_INTERVAL_SECONDS:
        batch_loss = loss.item()
        last_log = perf_counter()
        elapsed = last_log - training_start
        speed = processed / elapsed
        remaining = (len(subset) - processed) / speed
        print(
            f"Batch {batch:,}/{len(train):,} | {processed / len(subset):.1%} | "
            f"{processed:,} patches | batch loss {batch_loss:.4f} | {speed:.1f} patches/s | "
            f"elapsed {timedelta(seconds=int(elapsed))} | ETA {timedelta(seconds=round(remaining))}",
            flush=True,
        )

print(f"Saving checkpoint to {MODEL}...", flush=True)
MODEL.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL)
print(f"Saved {MODEL} | total time {timedelta(seconds=int(perf_counter() - started))}", flush=True)
