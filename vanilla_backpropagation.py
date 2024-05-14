import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

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
image.requires_grad = True
outputs = model(image)
loss = outputs[0][0]
loss.backward()

# Visualize the gradients
gradients = image.grad
gradients = gradients.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min())
plt.imshow(gradients, cmap='seismic')
plt.colorbar()
plt.title('Vanilla Backpropagation')
plt.show()