import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("Loading dataset...")
dataset = ImageFolder(root='./data/101_ObjectCategories', transform=transform)

# Split dataset

def split_dataset(dataset, train_size=30):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    train_indices, test_indices = [], []
    for indices in class_indices.values():
        np.random.shuffle(indices)
        train_indices += indices[:train_size]
        test_indices += indices[train_size:]

    return Subset(dataset, train_indices), Subset(dataset, test_indices)

train_set, val_set = split_dataset(dataset)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# Model setup: randomly initialized ResNet-18
model = models.resnet18(weights=None)  # No pretraining
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)

# All parameters require grad
for param in model.parameters():
    param.requires_grad = True

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Loss function
criterion = nn.CrossEntropyLoss()

# TensorBoard
writer = SummaryWriter(log_dir='runs/caltech101_scratch')

# Train and evaluate functions

def train(model, loader, optimizer, criterion, epoch, writer, tag):
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    writer.add_scalar(f'{tag}/train_loss', avg_loss, epoch)


def evaluate(model, loader, criterion, epoch, writer, tag):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()

    acc = correct / total
    writer.add_scalar(f'{tag}/val_loss', total_loss / len(loader), epoch)
    writer.add_scalar(f'{tag}/val_acc', acc, epoch)
    return acc

# Training loop
for epoch in range(30):
    train(model, train_loader, optimizer, criterion, epoch, writer, "scratch")
    acc = evaluate(model, val_loader, criterion, epoch, writer, "scratch")
    print(f"[Epoch {epoch}] Accuracy: {acc:.4f}")

# Optional: save the model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/resnet18_caltech101_scratch.pth")



