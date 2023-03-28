import torch
import torch.nn as nn
from torchvision import datasets, transforms

from src.models.DIFF_models import MNISTDiffusion
from src.trainers.trainDIFF import train


def run(batch_size, timesteps, lr):
    # Set device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Define Loss function
    criterion = nn.KLDivLoss(reduction="batchmean").to(device)

    # Beta range
    beta_zero = 0.02
    beta_last = 0.0001

    string_model = f"""
    DEVICE = {device}
    timesteps = {timesteps}
    batch_size = {batch_size}
    learning_rate = {lr}
    loss = {criterion}
    """

    print(string_model)

    # Define transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert PIL image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # Normalize the tensor with mean=0.1307 and std=0.3081
    ])

    # Download the training set and apply the transformations
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', train=True, download=True, transform=transform)

    # Create a DataLoader to iterate over the training set in batches
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Call model
    model = MNISTDiffusion(img_size=(28,28), timesteps=timesteps).to(device)

    # Define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    return train(model, device=device, training_dataset=trainloader, optimizer=optimizer, loss_function=criterion, times=timesteps, beta_zero=beta_zero, beta_end=beta_last)