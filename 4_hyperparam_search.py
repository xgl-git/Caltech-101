import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

def train_and_evaluate(train_loader, val_loader, lr, num_epochs, device):
    print(f"\n🔧 训练配置：lr={lr}, epochs={num_epochs}")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 102)
    model = model.to(device)

    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': lr},
        {'params': [param for name, param in model.named_parameters() if "fc" not in name], 'lr': lr * 0.1}
    ])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1}: loss = {avg_loss:.4f}")

    # 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"✅ 结果: lr={lr}, epochs={num_epochs} → 验证准确率 = {acc:.4f}")
    return acc

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数组合
    learning_rates = [1e-3, 5e-4, 1e-4]
    num_epochs_list = [10, 20]

    # 数据处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    data_dir = './data/101_ObjectCategories'
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 训练验证集划分
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 遍历组合训练
    results = {}
    for lr in learning_rates:
        for num_epochs in num_epochs_list:
            acc = train_and_evaluate(train_loader, val_loader, lr, num_epochs, device)
            results[(lr, num_epochs)] = acc

    # 输出排序结果
    print("\n🏆 超参数组合验证准确率排序：")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for (lr, ep), acc in sorted_results:
        print(f"lr={lr}, epochs={ep} → val_acc={acc:.4f}")

if __name__ == '__main__':
    main()
