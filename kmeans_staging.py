import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn

# === STEP 1: Wrap malignant images into dummy class folder ===
original_dir = r'data/train/malignant'
wrapped_dir = r'data/malignant_wrapped/dummy_class'

os.makedirs(wrapped_dir, exist_ok=True)

# Copy all images to the dummy_class folder
for file in os.listdir(original_dir):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        shutil.copy(os.path.join(original_dir, file), os.path.join(wrapped_dir, file))

# === STEP 2: Define transform and dataset ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder(root='data/malignant_wrapped', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# === STEP 3: Load ResNet18 and remove classifier ===
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet.eval()

# === STEP 4: Extract features ===
features = []
image_paths = []

with torch.no_grad():
    for inputs, _ in dataloader:
        outputs = resnet(inputs)
        features.append(outputs.numpy())
        batch_paths = [dataset.samples[i][0] for i in range(len(image_paths), len(image_paths) + len(inputs))]
        image_paths.extend(batch_paths)

features = np.vstack(features)

# === STEP 5: KMeans Clustering (pseudo-stage prediction) ===
kmeans = KMeans(n_clusters=4, random_state=42)
stage_labels = kmeans.fit_predict(features)

# === STEP 6: Create stage-wise folders and copy images ===
for path, label in zip(image_paths, stage_labels):
    stage_dir = f'data_stage/train/stage{label + 1}'
    os.makedirs(stage_dir, exist_ok=True)
    shutil.copy(path, os.path.join(stage_dir, os.path.basename(path)))

print("Pseudo-staging complete! Images are saved in: data_stage/train/stage1 to stage4")
