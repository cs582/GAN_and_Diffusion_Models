import torch
import torch.nn as nn
from torchvision import datasets, transforms

from src.models.generators.CNNs import MNISTGenerator
from src.models.classifiers.CNNs import MicroCNN

from src.trainers.trainGAN import train


def run(epochs, batch_size, latent_vector_size, lr):
    # Set device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Define Loss function
    criterion = nn.BCELoss().to(device)

    string_model = f"""
    DEVICE = {device}
    batch_size = {batch_size}
    epochs = {epochs}
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Call model
    generator = MNISTGenerator(in_size=latent_vector_size, out_shape=(28,28)).to(device)
    discriminator = MicroCNN(in_channels=1, out_size=1).to(device)

    # Define your optimizer
    G_optimizer = torch.optim.SGD(generator.parameters(), lr=lr)
    D_optimizer = torch.optim.SGD(discriminator.parameters(), lr=lr)

    train(generator, discriminator, device=device, latent_vector_size=latent_vector_size, training_dataset=trainloader, epochs=epochs, D_optimizer=D_optimizer, G_optimizer=G_optimizer, loss_function=criterion)