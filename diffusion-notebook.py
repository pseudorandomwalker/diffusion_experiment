# %% Cell 1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Using device: {device}')

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# %% Cell 2
# Load the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split into training and validation
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_subset, val_subset = random_split(trainset, [train_size, val_size])

trainloader = DataLoader(train_subset, batch_size=128, shuffle=True)
valloader = DataLoader(val_subset, batch_size=128, shuffle=False)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

classes = list(range(10))
logger.info('Data loaded and split into training and validation sets successfully')


# %% Cell 3
# Define the network architecture

class DiffusionModel(nn.Module):
    def __init__(self, label_dimensions):
        super(DiffusionModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)    # Input channels:1, Output:32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # Input:32, Output:64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Input:64, Output:128
        self.pool = nn.MaxPool2d(2, 2)                              # Reduces spatial dimensions by half
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3 + label_dimensions + 1, 1024)  # Adjust input size based on conv output
        self.reg1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.reg2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 28 * 28)  # Output: Flattened image (784 pixels)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x_noisy, labels, t):
        # Convolutional layers with activation and pooling
        x = F.leaky_relu(self.batchnorm1(self.conv1(x_noisy)))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.leaky_relu(self.batchnorm3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Flatten the feature maps
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 128*3*3)

        # Concatenate labels and time
        x = torch.cat((x, labels, t), dim=1)  # Shape: (batch_size, 128*3*3 + label_dimensions + 1)

        # Fully connected layers with activation and batch normalization
        x = F.leaky_relu(self.fc1(x))
        x = self.reg1(x)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.reg2(x)
        x = self.dropout(x)
        x = self.fc3(x)  # No activation on the output layer

        return x


# %% Cell 4
# Define the training loop
VARIANCE = 0.1
dt = 0.01
patience = 5  # For early stopping

def train(model, trainloader, valloader, criterion, optimizer, scheduler, epochs=20, patience=5):
    model.train()
    model.to(device)
    best_val_loss = float('inf')
    trigger_times = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = F.one_hot(labels, num_classes=10).float().to(device)
            inputs = inputs.to(device)

            # Sample random time steps t uniformly from [0,1]
            t = torch.rand(inputs.shape[0], 1, device=device)

            # Generate noise
            std = torch.sqrt(VARIANCE * t).view(t.size(0), 1, 1, 1).expand_as(inputs)  # Reshape and expand
            noise = torch.randn_like(inputs) * std  # Correct noise generation

            # Noisy input
            x_noisy = inputs + noise

            # Forward pass: predict the noise
            predicted_noise = model(x_noisy, labels, t.view(t.size(0), -1))  # Pass t as [batch,1]

            # Compute loss between predicted noise and actual noise
            loss = criterion(predicted_noise, noise.view(noise.size(0), -1))
            
            
            if i % 100 == 99:
            # Compare first sample in the batch
                predicted = predicted_noise[0].cpu().detach().numpy()
                actual = noise.view(noise.size(0), -1)[0].cpu().numpy()

                plt.figure(figsize=(10, 4))

                plt.subplot(1, 2, 1)
                plt.title("Predicted Noise")
                plt.plot(predicted[:100])  # Plot first 100 values

                plt.subplot(1, 2, 2)
                plt.title("Actual Noise")
                plt.plot(actual[:100])  # Plot first 100 values

                plt.show()



            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                avg_loss = running_loss / 100
                logger.info(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {avg_loss:.6f}')
                running_loss = 0.0

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(trainloader) if len(trainloader) > 0 else 0
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                labels = F.one_hot(labels, num_classes=10).float().to(device)
                inputs = inputs.to(device)

                # Sample random time steps t uniformly from [0,1]
                t = torch.rand(inputs.shape[0], 1, device=device)

                # Generate noise
                std = torch.sqrt(VARIANCE * t).view(t.size(0), 1, 1, 1).expand_as(inputs)
                noise = torch.randn_like(inputs) * std

                # Noisy input
                x_noisy = inputs + noise

                # Forward pass: predict the noise
                predicted_noise = model(x_noisy, labels, t.view(t.size(0), -1))  # Pass t as [batch,1]

                # Compute loss between predicted noise and actual noise
                loss = criterion(predicted_noise, noise.view(noise.size(0), -1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valloader)
        val_losses.append(avg_val_loss)
        logger.info(f'Epoch: {epoch + 1}, Validation Loss: {avg_val_loss:.6f}')

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info('Validation loss decreased. Saving model...')
        else:
            trigger_times += 1
            logger.info(f'No improvement in validation loss for {trigger_times} epochs.')

            if trigger_times >= patience:
                logger.info('Early stopping triggered!')
                break

        # Step the scheduler
        scheduler.step()

        # Switch back to training mode
        model.train()

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

    logger.info('Finished Training')



# %% Cell 5
# Define the model, criterion, optimizer, and scheduler
label_dimensions = 10  # One-hot encoded labels for digits 0-9

model = DiffusionModel(label_dimensions=label_dimensions).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

logger.info('Model, criterion, optimizer, and scheduler have been defined.')


# %% Cell 6
# Train the model
epochs = 50
patience = 5

train(model, trainloader, valloader, criterion, optimizer, scheduler, epochs=epochs, patience=patience)

# %% Cell 7
# Define the sampling function
def sample(model, label_idx, steps=int(1/dt), device='cpu'):
    model.eval()
    deltat = dt/10
    step_count = int(1/deltat)
    with torch.no_grad():
        # Initialize with pure noise
        x = torch.randn(1, 1, 28, 28, device=device) * np.sqrt(VARIANCE)
        label = F.one_hot(torch.tensor([label_idx], device=device), num_classes=10).float()
        samples = [x.cpu().numpy().squeeze()]

        for step in range(step_count):
            t = 1 - deltat * step
            t_tensor = torch.tensor([[t]], device=device)

            # Predict the noise
            predicted_noise = model(x, label, t_tensor.view(t_tensor.size(0), -1))  # Pass t as [batch,1]

            # Update the image by subtracting the predicted noise scaled by sqrt(VARIANCE * dt)
            scaling_factor = torch.sqrt(torch.tensor(VARIANCE * dt, device=device))
            x = x - predicted_noise.view(x.size())

            samples.append(x.cpu().numpy().squeeze())

    return samples

# %% Cell 8
# Sample from the model for a single label and plot denoising progression in a grid

def plot_denoising_progression(samples, label_idx, dt, max_cols=10):
    num_steps = len(samples)
    num_cols = int(min(max_cols, num_steps))
    num_rows = int(np.ceil(num_steps / num_cols))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    fig.suptitle(f'Denoising Progression for Label {label_idx}', fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        if i < num_steps:
            img = samples[i]
            img = img * 0.5 + 0.5  # Undo normalization
            img = np.clip(img, 0, 1)  # Ensure pixel values are in [0,1]
            ax.imshow(img, cmap='gray')
            ax.set_title(f't={1 - dt * i:.2f}')
            ax.axis('off')
        else:
            ax.axis('off')  # Hide unused subplots
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust top to accommodate the suptitle
    plt.show()

# Usage Example:
model.eval()
label_idx = 5  # For example, label '0'
samples = sample(model, label_idx, steps=int(1/dt), device=device)

plot_denoising_progression(samples, label_idx, dt, max_cols=10)



# %% Cell 9
# Generate samples for all labels
labels = torch.eye(10)  # One-hot vectors for digits 0-9
samples_dict = {i: [] for i in range(10)}

for label_idx in range(10):
    # Initialize with pure noise
    x = torch.randn(1, 1, 28, 28, device=device) * np.sqrt(VARIANCE)
    label = F.one_hot(torch.tensor([label_idx], device=device), num_classes=10).float()
    samples_dict[label_idx].append(x.cpu().numpy().squeeze())
    
    for step in range(int(1/dt)):
        t = 1 - dt * step
        t_tensor = torch.tensor([[t]], device=device)

        # Predict the noise
        predicted_noise = model(x, label, t_tensor)

        # Update the image by subtracting the predicted noise scaled by sqrt(VARIANCE * dt)
        x = x - torch.sqrt(torch.tensor(VARIANCE * dt, device=device)) * predicted_noise.view(x.size())
        samples_dict[label_idx].append(x.cpu().detach().numpy().squeeze())



# %% Cell 10
import math

# Number of labels
num_labels = 10
# Number of images per label to plot (e.g., final image)
images_per_label = 1  # Set to 1 to plot only the final image

# Calculate total number of images to plot
num_samples = num_labels * images_per_label
grid_size = math.ceil(math.sqrt(num_samples))

fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))

# Flatten the axes for easy iteration
axes = axes.flatten()

# Counter for plotted images
plot_count = 0

for label_idx in range(num_labels):
    for img_idx in range(images_per_label):
        if plot_count >= len(axes):
            break  # Prevent index out of range if grid is smaller
        
        ax = axes[plot_count]
        
        # Select the specific image to plot
        # For example, the last image (final generated image)
        img = samples_dict[label_idx][-1]  # Change -1 to img_idx if you have multiple images per label
        
        # Undo normalization
        img = img * 0.5 + 0.5  
        img = np.clip(img, 0, 1)  # Ensure pixel values are in [0,1]
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Label: {label_idx}")
        
        plot_count += 1

# Hide any unused subplots
for i in range(plot_count, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# %% Cell 11
# Visualize intermediate denoising steps for each label
for label_idx in range(10):
    imgs = samples_dict[label_idx]
    num_steps = len(imgs)
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
    fig.suptitle(f'Denoising Progression for Label {label_idx}', fontsize=16)
    
    for i, ax in enumerate(axes):
        img = imgs[i]
        img = img * 0.5 + 0.5  # Undo normalization
        img = np.clip(img, 0, 1)  # Ensure pixel values are in [0,1]
        ax.imshow(img, cmap='gray')
        ax.set_title(f't={1 - dt * i:.2f}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
