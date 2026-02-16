# overfit_debug.py (place in examples/CIFAR-100/)
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from cifar100_vit import MODEL
from cifar100_base import TRAIN_DATASET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MODEL.to(device)
subset = Subset(TRAIN_DATASET, list(range(128)))   # tiny subset
loader = DataLoader(subset, batch_size=32, shuffle=True)
opt = optim.Adam(model.parameters(), lr=3e-4)
crit = torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        out = model(data)
        loss = crit(out, target)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} Loss {total_loss/len(loader):.4f} Acc {correct/total:.4f}")