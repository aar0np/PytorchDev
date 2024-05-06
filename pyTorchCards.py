import matplotlib.pyplot as plt # visualization
import numpy as np
import os
import pandas as pd             # dataframes
import sys
import timm                     # deep learning library
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from matplotlib import image as mpimg
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('Numpy:', np.__version__)
print('Pandas:', pd.__version__)

class PlayingCardDataset(Dataset):
	def __init__(self, data_dir, transform=None):
		self.data = ImageFolder(data_dir, transform=transform)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		return self.data[index]

	@property
	def classes(self):
		return self.data.classes

#data_dir = '/kaggle/input/cards-image-datasetclassification/train'
data_dir = '~/local/playing_card_images_dataset/train'

dataset = PlayingCardDataset(
	data_dir = data_dir
)

print(f"Dataset len == {len(dataset)}")

#print(dataset[0])
image, label = dataset[6000]
print(f"label == {label}")
plt.imshow(image)
plt.show()

# get a dictionary mapping target values with folder names
target_to_class = {v: k for k,v in ImageFolder(data_dir).class_to_idx.items()}
print(target_to_class)

# add transformer and recreate dataset and keep all images the same size
transform = transforms.Compose([
	transforms.Resize((128,128)),
	transforms.ToTensor(),
])

dataset = PlayingCardDataset(data_dir,transform)

image, label = dataset[100]
print(f"{image.shape}")

# iterate over dataset
#for image, label in dataset:
#	break

# create Pytorch dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# iterate over dataloader
for images, labels in dataloader:
	print(f"{image.shape}")
	break

# 2.0 - create Pytorch model
class SimpleCardClassifier(nn.Module):
	def __init__(self, num_classes=53):
		super(SimpleCardClassifier, self).__init__()
		# define all parts of our model
		self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
		self.features = nn.Sequential(*list(self.base_model.children())[:-1])

		enet_out_size = 1280
		# create a classifier
		self.classifier = nn.Linear(enet_out_size, num_classes)

	def forward(self, x):
		#connect parts, return the model
		x = self.features(x)
		output = self.classifier(x)
		return output

model = SimpleCardClassifier(num_classes=53)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_file = Path("playing_card_model.pyt")
if model_file.is_file() == False:

	# show last 500 chars of model
	print(str(model)[:500])

	example_out = model(images)
	print(example_out)
	print(example_out.shape) # [batch_size, num_classes]

	# 3.0 - train model
		# needs optimizer and loss function

	# Loss function
	criterion = nn.CrossEntropyLoss()
	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=0.001) 

	# model hasn't learned anything, so loss will be high (~3.9-4.0)
	print(criterion(example_out, labels))

	transform = transforms.Compose([
		transforms.Resize((128, 128)),
		transforms.ToTensor(),
	])

	train_folder = '~/local/playing_card_images_dataset/train/'
	validation_folder = '~/local/playing_card_images_dataset/valid/'
	test_folder = '~/local/playing_card_images_dataset/test/'

	train_dataset = PlayingCardDataset(train_folder, transform=transform)
	validation_dataset = PlayingCardDataset(train_folder, transform=transform)
	test_dataset = PlayingCardDataset(train_folder, transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

	num_epoch = 5
	train_losses, validation_losses = [], []

	model = SimpleCardClassifier(num_classes=53)
	model.to(device)

	for epoch in range(num_epoch):
		# set model to training mode
		model.train()
		running_loss = 0.0
		
		for images,labels in tqdm(train_loader, desc='Training loop'):
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * images.size(0)

		train_loss = running_loss / len(train_loader.dataset)
		train_losses.append(train_loss)

		# validation phase
		# set model to evaluation mode
		model.eval()
		running_loss = 0.0
		with torch.no_grad():
			for images, labels in tqdm(validation_loader, desc='Validation loop'):
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)
				loss = criterion(outputs, labels)
				running_loss += loss.item() + images.size(0)

		validation_loss = running_loss / len(validation_loader.dataset)
		validation_losses.append(validation_loss)

		# print epoch stats
		print(f"Epoch {epoch+1}/{num_epoch} - Training loss: {train_losses}, Validation loss: {validation_losses}")

	# save our model
	torch.save(model.state_dict(), "playing_card_model.pyt")

	plt.plot(train_losses, label='Training loss')
	plt.plot(validation_losses, label='Validation loss')
	plt.legend()
	plt.title("Loss over epochs")
	plt.show()
else:
	# model file exists, load
	model.load_state_dict(torch.load("playing_card_model.pyt"))

# BONUS - evaluating results, visualization

# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

# Visualization
def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
    
    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    
    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

# Example usage
test_image = "/Users/aaron.ploetz/local/playing_card_images_dataset/test/five of diamonds/2.jpg"
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

original_image, image_tensor = preprocess_image(test_image, transform)
probabilities = predict(model, image_tensor, device)

# Assuming dataset.classes gives the class names
class_names = dataset.classes 
visualize_predictions(original_image, probabilities, class_names)
