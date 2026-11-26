import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from settings import BATCH_SIZE, load_camelyon17, MODEL, SEED, image_transform

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
data = load_camelyon17()
train = DataLoader(data.get_subset("train", transform=image_transform()), batch_size=BATCH_SIZE, shuffle=True)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.requires_grad_(False)
model.fc = nn.Linear(model.fc.in_features, 2)
# Keep the pretrained batch-normalization statistics fixed.
model.to(device).eval()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for images, labels, _ in train:
    optimizer.zero_grad()
    logits = model(images.to(device))
    loss = loss_fn(logits, labels.to(device))
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), MODEL)
print("Saved", MODEL)
