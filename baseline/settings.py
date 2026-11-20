from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "baseline/data"
MODEL = ROOT / "models/resnet18_linear_head.pt"
BATCH_SIZE = 64
SEED = 42


def image_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
