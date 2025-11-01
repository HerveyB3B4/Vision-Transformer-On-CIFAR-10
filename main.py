import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

from VisionTransformer import VisionTransformer


class DatasetWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: nn.Module,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(
    model: nn.Module, loader: DataLoader, loss_function: nn.Module, device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(loader)


def test_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    # 定义
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),  # 平移
            transforms.RandomHorizontalFlip(),  # 翻转
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_validation_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # 加载训练数据
    full_train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=None
    )

    # 分割训练集和校验集
    train_size = int(0.9 * len(full_train_dataset))
    validation_size = len(full_train_dataset) - train_size
    train_dataset_tmp, validation_dataset_tmp = random_split(
        full_train_dataset, [train_size, validation_size]
    )
    train_dataset = DatasetWrapper(train_dataset_tmp, transform=transform_train)
    validation_dataset = DatasetWrapper(
        validation_dataset_tmp, transform=transform_validation_test
    )

    # 加载测试数据
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_validation_test
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=128, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 实例化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    model = VisionTransformer(
        image_size=32,
        in_channels=3,
        patch_size=4,
        embedding_dimensions=512,
        heads_num=8,
        mlp_dimensions=2048,
        dropout=0.1,
        transformer_layers_num=6,
        classes_num=10,
    ).to(device)

    # 定义损失函数和优化器
    LossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, optimizer, LossFunction, device)
        validation_loss = validate_epoch(model, validation_loader, LossFunction, device)

        print(
            f"Epoch [{epoch+1}/50], Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}"
        )

    test_accuracy = test_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
