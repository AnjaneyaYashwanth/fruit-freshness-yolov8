import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset_loader import DualStreamDataset
from models.dual_stream_model import DualStreamModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Dataset
# ----------------------------
train_dataset = DualStreamDataset("data/processed", "train")
val_dataset = DualStreamDataset("data/processed", "val")

# ----------------------------
# GWO Settings
# ----------------------------
num_wolves = 5
num_iterations = 8

# Focused search ranges
lr_range = [0.0003, 0.0006]
dropout_range = [0.2, 0.4]
batch_options = [4, 8]

# ----------------------------
# Fitness Function
# ----------------------------
def fitness(lr, dropout, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = DualStreamModel(dropout=dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 🔥 fast training
    for epoch in range(4):
        model.train()
        for rgb, edge, lbp, labels in train_loader:
            rgb, edge, lbp, labels = rgb.to(device), edge.to(device), lbp.to(device), labels.to(device)

            outputs = model(rgb, edge, lbp)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for rgb, edge, lbp, labels in val_loader:
            rgb, edge, lbp, labels = rgb.to(device), edge.to(device), lbp.to(device), labels.to(device)

            outputs = model(rgb, edge, lbp)
            _, pred = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return correct / total


# ----------------------------
# Initialize Wolves
# ----------------------------
wolves = []

for _ in range(num_wolves):
    wolf = {
        "lr": np.random.uniform(*lr_range),
        "dropout": np.random.uniform(*dropout_range),
        "batch": int(np.random.choice(batch_options))
    }
    wolves.append(wolf)

# ----------------------------
# GWO Optimization Loop
# ----------------------------
best_wolf = None
best_score = 0

for iteration in range(num_iterations):
    print(f"\n🔥 Dual-Stream GWO Iteration {iteration+1}")

    for wolf in wolves:
        score = fitness(wolf["lr"], wolf["dropout"], wolf["batch"])

        print(f"LR: {wolf['lr']:.6f}, Dropout: {wolf['dropout']:.2f}, Batch: {wolf['batch']} → Acc: {score:.4f}")

        if score > best_score:
            best_score = score
            best_wolf = wolf

    # Update wolves around best
    for wolf in wolves:
        wolf["lr"] = np.clip(best_wolf["lr"] + np.random.uniform(-0.0001, 0.0001), *lr_range)
        wolf["dropout"] = np.clip(best_wolf["dropout"] + np.random.uniform(-0.05, 0.05), *dropout_range)
        wolf["batch"] = int(np.random.choice(batch_options))

print("\n🏆 BEST PARAMETERS (Dual-Stream):")
print(best_wolf)
print(f"Best Accuracy: {best_score:.4f}")