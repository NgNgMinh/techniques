import os
from sklearn.model_selection import train_test_split
import shutil

# Set paths
data_dir = 'data'
dog_dir = os.path.join(data_dir, 'dogs')
cat_dir = os.path.join(data_dir, 'cats')

# Get image paths
dog_images = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir)]
cat_images = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir)]

# Split into train and val sets
dog_train, dog_val, cat_train, cat_val = train_test_split(dog_images, cat_images, test_size=0.2, random_state=42)

# Create new directories
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

train_dog_dir = os.path.join(train_dir, 'dogs')
train_cat_dir = os.path.join(train_dir, 'cats')
os.makedirs(train_dog_dir, exist_ok=True)
os.makedirs(train_cat_dir, exist_ok=True)

val_dog_dir = os.path.join(val_dir, 'dogs')
val_cat_dir = os.path.join(val_dir, 'cats')
os.makedirs(val_dog_dir, exist_ok=True)
os.makedirs(val_cat_dir, exist_ok=True)

# Copy images to new directories
for img in dog_train:
    shutil.copy(img, train_dog_dir)
for img in cat_train:
    shutil.copy(img, train_cat_dir)
for img in dog_val:
    shutil.copy(img, val_dog_dir)
for img in cat_val:
    shutil.copy(img, val_cat_dir)