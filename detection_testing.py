import torch
from torchvision import models, transforms
from PIL import Image

# === Labels ===
class_labels = ['benign', 'malignant']
stage_labels = ['stage1', 'stage2', 'stage3', 'stage4']

# === Transform ===++
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Load Classifier Model ===
clf_model = models.resnet18(pretrained=False)
clf_model.fc = torch.nn.Linear(clf_model.fc.in_features, 2)
clf_model.load_state_dict(torch.load('cancer_model.pth'))
clf_model.eval()

# === Load Stage Model ===
stage_model = models.resnet18(pretrained=False)
stage_model.fc = torch.nn.Linear(stage_model.fc.in_features, 4)
stage_model.load_state_dict(torch.load('stage_classifier.pth'))
stage_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clf_model = clf_model.to(device)
stage_model = stage_model.to(device)

# === Prediction Function ===
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Step 1: Benign or Malignant
        output = clf_model(img_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        label = class_labels[pred_class]

        if label == 'malignant':
            stage_output = stage_model(img_tensor)
            stage = stage_labels[torch.argmax(stage_output, dim=1).item()]
            print(f"🔬 Result: Malignant — Stage: {stage}")
        else:
            print("✅ Result: Benign")

# === Test It ===
predict("testing image.jpg")  # Replace with your image filename
