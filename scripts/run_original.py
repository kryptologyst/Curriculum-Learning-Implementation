"""Original curriculum learning implementation for comparison."""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class SimpleCNN(nn.Module):
    """Original simple CNN implementation."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def curriculum_learning_step(data, step):
    """Original curriculum learning step implementation."""
    if step == 1:
        # "Easy" - No noise or occlusion
        return data
    elif step == 2:
        # "Medium" - Apply slight noise to images
        noise = torch.normal(mean=0, std=0.1, size=data.size())
        return data + noise
    elif step == 3:
        # "Hard" - Apply more occlusion to images
        data[:, :, 10:15, 10:15] = 0  # Black-out a region
        return data


def main():
    """Run the original curriculum learning implementation."""
    print("Running original curriculum learning implementation...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize the model, optimizer, and loss function
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    
    # Curriculum learning training loop
    for epoch in range(5):  # Training with curriculum for 5 epochs
        model.train()
        total_loss = 0
        for step, (data, target) in enumerate(train_loader, 1):
            # Apply curriculum learning step
            data = curriculum_learning_step(data, step=epoch % 3 + 1)  # Alternating curriculum steps (1, 2, 3)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    
    # Evaluate the model on the validation set
    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    
    return accuracy


if __name__ == "__main__":
    accuracy = main()
    print(f"\nOriginal implementation completed with {accuracy:.2f}% accuracy")
