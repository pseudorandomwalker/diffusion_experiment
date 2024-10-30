# %% Cell 1
import matplotlib
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# %% Cell 2
# Define the transformation for the data
# The transformation is used to convert the data to a tensor and normalize it to a range of [0, 1]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load the training data
# trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

# Download and load the test data
# testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

# Classes
# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print('Data loaded successfully')
print(f'Train data shape: {trainset.data.shape}')
print(f'Test data shape: {testset.data.shape}')
# %% Cell 3
# Define the network architecture
import torch.nn as nn
import torch.nn.functional as F

# %% Cell 4
# The model is a diffusion model with 5 layers
# The first layer is the input layer with the size of the input data, a flattened vector of the image and the label
# The last layer is the output layer with the size of the output data, also a flattened vector

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(28*28*3 + 10 + 1, 512)
        self.reg1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.reg2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.reg3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 512)
        self.reg4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 28*28*3)
        self.reg5 = nn.BatchNorm1d(28*28*3)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.reg1(x)
        x = F.tanh(self.fc2(x))
        x = self.reg2(x)
        x = F.tanh(self.fc3(x))
        x = self.reg3(x)
        x = F.tanh(self.fc4(x))
        x = self.reg4(x)
        x = F.tanh(self.fc5(x))
        return x
#

# class DiffusionModel(nn.Module):
#     def __init__(self):
#         super(DiffusionModel, self).__init__()
#         self.fc1 = nn.Conv2d(3, 16, 5)
#         self.reg1 = nn.BatchNorm2d(16)
#         self.fc2 = nn.Conv2d(16, 32, 5)
#         self.reg2 = nn.BatchNorm2d(32)
#         self.fc3 = nn.Conv2d(32, 64, 5)
#         self.reg3 = nn.BatchNorm2d(64)
#         self.fc4 = nn.Linear(64 + 10 + 1, 512)
#         self.reg4 = nn.BatchNorm1d(512)
#         self.fc5 = nn.Linear(512, 28*28*3)

#     def forward(self, x, other_info):
#         x = F.tanh(self.fc1(x))
#         x = self.reg1(x)
#         x = F.tanh(self.fc2(x))
#         x = self.reg2(x)
#         x = F.tanh(self.fc3(x))
#         x = self.reg3(x)
#         x = torch.cat((x.view(x.shape[0], -1), other_info), 1)
#         x = F.tanh(self.fc4(x))
#         x = self.reg4(x)
#         x = F.tanh(self.fc5(x))
#         return x

# %% Cell 5
model = DiffusionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('Model created successfully')
# %% Cell 6
# Train the model
import matplotlib.pyplot as plt

epochs = 1
terminal_variance = 0.1
delta_t = 0.001
epoch_losses = []
batch_losses = []

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # One hot encode the labels
        labels = F.one_hot(labels, num_classes=10).float()

        # Flatten the input data
        # inputs = inputs.view(inputs.shape[0], -1)
        # print(inputs.shape, labels.shape)

        # Add noise to the input data
        # The noise is a random sample from a normal distribution with mean 0 and variance terminal_variance
        # The noise is added to the input data and the data is diffused
        tees = torch.rand(inputs.shape[0], 1)
        tsteps_random_samples = torch.normal(0.0, terminal_variance ** 2 * tees.expand(inputs.shape[0], inputs.shape[1], inputs.shape[2]))
        delta_tsteps_random_samples = torch.normal(0.0, terminal_variance ** 2 * delta_t, size=(inputs.shape[0], inputs.shape[1], inputs.shape[2]))

        initial_diffused_inputs = inputs + tsteps_random_samples
        delta_t_diffused_inputs = initial_diffused_inputs + delta_tsteps_random_samples

        model_input = torch.cat((delta_t_diffused_inputs, labels, tees + delta_t), dim=1)

        optimizer.zero_grad()
        model_output = model(model_input)
        loss = criterion(model_output, initial_diffused_inputs)

        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

        running_loss += loss.item()
        # Print batch loss every 100 batches
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    epoch_losses.append(sum(batch_losses[-len(trainloader):]) / len(trainloader))

# %% Cell 7

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(batch_losses)
plt.title('Batch Loss over Training')
plt.xlabel('Batch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(epoch_losses)
plt.title('Average Epoch Loss over Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# %% Cell 8
# Sample the model
# Sample the model by generating a random sample from a normal distribution with mean 0 and variance 1
# The sample is then passed through the model to generate a new sample
# The new sample is then passed through the model again to generate a new sample
# This process is repeated for a number of steps to generate a sequence of samples
#
# The sequence of samples is then plotted to show the diffusion process
# The diffusion process is the process of generating a sequence of samples from a model

# Generate a random sample
model.eval()
sample = torch.normal(0.0, terminal_variance**2, size=(1, 28*28*3))
sample = sample.view(1, -1)
label = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]).float()
# label = F.one_hot(torch.tensor(0), num_classes=10).float()
sample = torch.cat((sample, label, torch.tensor([[terminal_variance]])), dim=1)
time = terminal_variance

# print(sample.shape)
# print(model(sample).shape)
# # Generate a sequence of samples
samples = []
# for i in range(1, int(1/terminal_variance) + 1):
while time > 0:
    new_sample = model(sample)
    samples.append(new_sample.view(3, 28, 28).detach().permute(1, 2, 0).numpy())
    new_sample += torch.normal(0.0, terminal_variance**2 * 0.01, size=(1, 28*28*3))
    sample = torch.cat((new_sample, label, torch.tensor([[terminal_variance - delta_t]])), dim=1)
    time -= delta_t


# %% Cell 9
# Plot the diffusion process

fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    # Denormalize the sample
    img = samples[i] * 0.5 + 0.5  # Undo the normalization
    img += img.min()
    img /= img.max()
    # if i % 5 == 0:
        # print(img[0, 0, :])
    # img = samples[i]
    print(img.min(), img.max())
    ax.imshow(img)
    ax.axis('off')

plt.suptitle('Diffusion Process')
plt.show()



# %% Cell 10

dataiter = iter(trainloader)
images, labels = next(dataiter)

# Plot the images
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    # Denormalize the image
    #
    img = images[i]
    print(Z)
    img = img.permute(1, 2, 0).numpy() * 0.5 + 0.5
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Label: {labels[i].item()}')

plt.suptitle('Sample Images from Train Loader')
plt.tight_layout()
plt.show()

# %% Cell 11
print(len(samples))
