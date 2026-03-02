import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random


# we is making random seeds same
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# downloading the clothes datas
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# taking only one batchs for testing
single_batch = next(iter(trainloader))
single_batch_loader = [(single_batch[0], single_batch[1])]


# this network are very simple
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x): return self.net(x)


# this network have more layers
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x): return self.net(x)


# dropping out some neurons so it not memorize
class RegularizedComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x): return self.net(x)


# loop for learn the weights
def train_model(model, loader, val_loader, epochs, lr=0.001, weight_decay=0.0):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total

        # checking on valid datas
        val_acc = 0
        if val_loader:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for v_inputs, v_targets in val_loader:
                    v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                    v_outputs = model(v_inputs)
                    _, v_predicted = v_outputs.max(1)
                    val_total += v_targets.size(0)
                    val_correct += v_predicted.eq(v_targets).sum().item()
            val_acc = 100. * val_correct / val_total

        if (epoch + 1) == epochs or (epoch + 1) % 5 == 0:
            val_str = f" | Val Acc: {val_acc:.2f}%" if val_loader else ""
            print(
                f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(loader):.4f} | Train Acc: {train_acc:.2f}%{val_str}")


if __name__ == "__main__":
    print("==================================================")
    print("STEP 1: SANITY CHECK (Code Verification)")
    print("Goal: Overfit a single batch perfectly. Loss should approach 0.")
    print("==================================================")
    sanity_model = SimpleModel()
    train_model(sanity_model, single_batch_loader, val_loader=None, epochs=50)

    print("\n==================================================")
    print("STEP 2: ESTABLISH A BASELINE (Good Enough)")
    print("Goal: Train simple model on full data. Get a decent starting metric.")
    print("==================================================")
    baseline_model = SimpleModel()
    train_model(baseline_model, trainloader, testloader, epochs=5)

    print("\n==================================================")
    print("STEP 3: REDUCE BIAS (Fix Underfitting)")
    print("Goal: Make the model complex enough to overfit the training data.")
    print("Observation: Train Acc will be very high, Val Acc will lag behind.")
    print("==================================================")
    complex_model = ComplexModel()
    train_model(complex_model, trainloader, testloader, epochs=10)

    print("\n==================================================")
    print("STEP 4: REDUCE VARIANCE (Fix Overfitting - The Gold Standard)")
    print("Goal: Add Regularization (Dropout + L2 Weight Decay) to complex model.")
    print("Observation: Train Acc drops slightly, but Val Acc improves/stabilizes.")
    print("==================================================")
    reg_model = RegularizedComplexModel()
    train_model(reg_model, trainloader, testloader, epochs=10, weight_decay=1e-4)