import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transforms
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Load image
image = Image.open('data/test/4.jpg')
image = data_transforms(image)
image = image.unsqueeze(0)
image = image.to(device)

# Get the gradients of the image
layer_gc = LayerGradCam(model, model.layer4)
attributions = layer_gc.attribute(image, target=0)

# Visualize the gradients
attributions = attributions.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())
plt.imshow(attributions, cmap='seismic')
plt.colorbar()
plt.title('Grad-CAM')
plt.show()