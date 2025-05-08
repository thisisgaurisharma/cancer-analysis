import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score

# === Paths ===
train_dir = 'data_stage/train'

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Datasets and Loaders ===
dataset = datasets.ImageFolder(train_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# === Model: ResNet18 for 4-class classification ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# === Loss and Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# === Training Loop ===
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds += outputs.argmax(1).tolist()
        all_labels += labels.tolist()
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f} - Acc: {acc*100:.2f}%")

print("Stage classification training complete.")

torch.save(model.state_dict(), 'stage_classifier.pth')
