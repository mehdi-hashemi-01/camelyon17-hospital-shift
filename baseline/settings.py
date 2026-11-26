import warnings
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


def load_camelyon17():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^pkg_resources is deprecated as an API",
            category=UserWarning,
            module=r"^outdated$",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^The given NumPy array is not writable",
            category=UserWarning,
            module=r"^wilds\.datasets\.camelyon17_dataset$",
        )
        from wilds import get_dataset

        return get_dataset(dataset="camelyon17", root_dir=str(DATA), download=False)
