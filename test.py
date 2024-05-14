import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import glob
from PIL import Image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transforms
data_transforms = transforms.Compose({
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
})

# Load data
data_dir = 'data/test'
for path in glob.glob(data_dir + '/*'):
    image = Image.open(path)
    image = data_transforms(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    # Load model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        print('Cat' if preds.item() == 0 else 'Dog')

        img = cv2.imread(path)
        cv2.putText(img, 'Cat' if preds.item() == 0 else 'Dog', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('image', img)
        cv2.waitKey(0)