import csv

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from wilds import get_dataset

from settings import BATCH_SIZE, DATA, MODEL, ROOT, image_transform

device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet18(weights=None, num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL, map_location=device, weights_only=True))
model.eval()
data = get_dataset(dataset="camelyon17", root_dir=str(DATA), download=False)
summary = []
with torch.inference_mode():
    for split in ["id_val", "val"]:
        subset = data.get_subset(split, transform=image_transform())
        correct = 0
        confidence_sum = 0.0
        brier_sum = 0.0
        for images, labels, _ in DataLoader(subset, batch_size=BATCH_SIZE):
            probs = model(images.to(device)).softmax(dim=1).cpu()
            confidence, predictions = probs.max(dim=1)
            correct += (predictions == labels).sum().item()
            confidence_sum += confidence.sum().item()
            brier_sum += ((probs[:, 1] - labels) ** 2).sum().item()
        n = len(subset)
        summary.append([split, n, correct / n, confidence_sum / n, brier_sum / n])
        print(split, f"accuracy={correct / n:.4f}", f"brier={brier_sum / n:.4f}")

with (ROOT / "results/validation_metrics.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["split", "patches", "accuracy", "mean_confidence", "binary_brier"])
    writer.writerows(summary)
