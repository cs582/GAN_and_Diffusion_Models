import torch
import torch.nn as nn
from torchvision import datasets, transforms

from src.utils.transformations import noice_dataset
from src.models.Diffusion_models import MNISTDiffusion
from src.trainers.trainDiffusion import train


def run(batch_size, latent_vector_size, lr):
    # Set device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Define Loss function
    criterion = nn.KLDivLoss().to(device)

    string_model = f"""
    DEVICE = {device}
    batch_size = {batch_size}
    learning_rate = {lr}
    latent_vector_size = {latent_vector_size}
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
    pre_trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Noiced Images dataset
    trainloader = torch.utils.data.DataLoader(noice_dataset(pre_trainloader), batch_size=batch_size, shuffle=True)

    # Call model
    model = MNISTDiffusion(img_size=(28,28))

    # Define your optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train(model, device=device, training_dataset=trainloader, optimizer=optimizer, loss_function=criterion)