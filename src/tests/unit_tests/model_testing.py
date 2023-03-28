import unittest
import torch
import time

from src.models.backbones.CNNs import MicroCNN, MiniCNN
from src.models.blocks import SimpleCNNBlock, MLP, MultiscaleConvolution, DiffusionBlock, DiffusionDense
from src.models.GAN_models import MNISTGenerator
from src.models.DIFF_models import MNISTDiffusion


class TestBuildingBlocks(unittest.TestCase):
    def test_simplecnnblock_forward(self):
        # Print to console
        print("Testing SimpleCNNBlock Forward...")

        # Initialize the model and input tensor
        s = time.time()
        model = SimpleCNNBlock(in_channels=3, kernel_size=3, padding=0, stride=1, out_channels=6)
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")

        input_tensor = torch.randn(1, 3, 28, 28)

        # Run the forward pass and check the output
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 6, 26, 26)) # output should have shape (batch_size, num_classes)

        # Testing completed
        print(f"Testing SimpleCNNBlock COMPLETED")

    def test_mlp_forward(self):
        # Print to console
        print("Testing MLP Forward...")

        # Initialize the model and input tensor
        s = time.time()
        model = MLP(in_size=512, out_size=10)
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")
        input_tensor = torch.randn(1, 512)

        # Run the forward pass and check the output
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10)) # output should have shape (batch_size, num_classes)

        # Testing completed
        print("Testing MLP COMPLETED!!!")

    def test_mlp_backward(self):
        # Print to console
        print("Testing MLP Backward...")

        # Initialize the model and input/output tensors
        s = time.time()
        model = MLP(in_size=512, out_size=10)
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")

        input_tensor = torch.randn(1, 512)
        output_tensor = torch.randn(1, 10)

        # Compute the loss and gradients
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output_tensor, model(input_tensor))
        loss.backward()

        # Check that the gradients are not None
        for param in model.parameters():
            self.assertIsNotNone(param.grad)

        # Testing completed
        print("Testing MLP Backward COMPLETED!!!")


###########################################################
# Testing CNN models
###########################################################

class TestMicroCNN(unittest.TestCase):
    def test_forward(self):
        # Print to console
        print("Testing MicroCNN Forward...")

        # Initialize the model and input tensor
        s = time.time()
        model = MicroCNN(in_channels=1, out_size=10)
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")
        input_tensor = torch.randn(1, 1, 28, 28)

        # Run the forward pass and check the output
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10)) # output should have shape (batch_size, num_classes)

        # Testing completed
        print("Testing MicroCNN Forward COMPLETED!!!")

    def test_backward(self):
        # Print to console
        print("Testing MicroCNN Backward...")

        # Initialize the model and input/output tensors
        s = time.time()
        model = MicroCNN(in_channels=1, out_size=10)
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")
        input_tensor = torch.randn(1, 1, 28, 28)
        output_tensor = torch.randn(1, 10)

        # Compute the loss and gradients
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output_tensor, model(input_tensor))
        loss.backward()

        # Check that the gradients are not None
        for param in model.parameters():
            self.assertIsNotNone(param.grad)

        # Testing completed
        print("Testing MicroCNN Backward COMPLETED!!!")


class TestMiniCNN(unittest.TestCase):
    def test_forward(self):
        # Print to console
        print("Testing MiniCNN Forward...")

        # Initialize the model and input tensor
        s = time.time()
        model = MiniCNN(in_channels=3, out_size=10)
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")
        input_tensor = torch.randn(1, 3, 256, 256)

        # Run the forward pass and check the output
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10)) # output should have shape (batch_size, num_classes)

        # Testing completed
        print("Testing MiniCNN Forward COMPLETED!!!")

    def test_backward(self):
        # Print to console
        print("Testing MiniCNN Backward...")

        # Initialize the model and input/output tensors
        s = time.time()
        model = MiniCNN(in_channels=3, out_size=10)
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")
        input_tensor = torch.randn(1, 3, 256, 256)
        output_tensor = torch.randn(1, 10)

        # Compute the loss and gradients
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output_tensor, model(input_tensor))
        loss.backward()

        # Check that the gradients are not None
        for param in model.parameters():
            self.assertIsNotNone(param.grad)

        # Testing completed
        print("Testing MiniCNN Backward COMPLETED!!!")


###########################################################
# Testing Generative Models
###########################################################

class TestMNISTGen(unittest.TestCase):
    def test_forward(self):
        # Print to console
        print("Testing MNIST Generator Forward...")

        # Initialize the model and input tensor
        s = time.time()
        model = MNISTGenerator(in_size=10, out_shape=(28,28))
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")
        input_tensor = torch.randn(4, 10)

        # Run the forward pass and check the output
        output = model(input_tensor)
        self.assertEqual(output.shape, (4, 1, 28, 28)) # output should have shape (batch_size, num_classes)

        # Testing completed
        print("Testing MNIST Generator Forward COMPLETED!!!")

    def test_backward(self):
        # Print to console
        print("Testing MNIST Generator Backward...")

        # Initialize the model and input/output tensors
        s = time.time()
        model = MNISTGenerator(in_size=10, out_shape=(28,28))
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")
        input_tensor = torch.randn(4, 10)
        output_tensor = torch.randn(4, 1, 28, 28)

        # Compute the loss and gradients
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output_tensor, model(input_tensor))
        loss.backward()

        # Check that the gradients are not None
        for param in model.parameters():
            self.assertIsNotNone(param.grad)

        # Testing completed
        print("Testing MNIST Generator Backward COMPLETED!!!")


###########################################################
# Testing Diffusion Models
###########################################################

class TestDifussionModels(unittest.TestCase):
    def test_diffusion_MNIST_model(self):
        # Print to console
        print("Testing diffusion MNIST Forward...")

        # Initialize the model and input tensor
        T = 1000
        s = time.time()
        model = MNISTDiffusion(img_size=(64, 64), timesteps=T)
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")
        input_tensor = torch.randn(8, 1, 64, 64)

        # Run the forward pass and check the output
        output = model(input_tensor, torch.tensor([25]))
        self.assertEqual(output.shape, (8, 1, 64, 64))

        # Testing completed
        print("Testing diffusion MNIST Forward COMPLETED!!!")

    def test_multiscale_forward(self):
        # Print to console
        print("Testing Multiscale Convolution Forward...")

        # Initialize the model and input tensor
        s = time.time()
        model = MultiscaleConvolution(img_size=(256, 256), in_channels=3, kernel=1)
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")
        input_tensor = torch.randn(8, 3, 64, 64)

        # Run the forward pass and check the output
        output = model(input_tensor)
        self.assertEqual(output.shape, (8, 3, 64, 64)) # output should have shape (batch_size, num_classes)

        # Testing completed
        print("Testing Multiscale Convolution Forward COMPLETED!!!")

    def test_diffusion_dense_forward(self):
        # Print to console
        print("Testing Diffusion Dense Forward...")

        # Initialize the model and input tensor
        s = time.time()
        model = DiffusionDense(img_size=(28, 28), in_channels=1)
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")
        input_tensor = torch.randn(4, 1, 28, 28)

        # Run the forward pass and check the output
        output = model(input_tensor)
        self.assertEqual(output.shape, (4, 1, 28, 28)) # output should have shape (batch_size, num_classes)

        # Testing completed
        print("Testing Diffusion Dense Forward COMPLETED!!!")


    def test_diffusion_block_dense_forward(self):
        # Print to console
        print("Testing Diffusion Block Forward...")

        # Initialize the model and input tensor
        img_size = (32, 32)
        s = time.time()
        model = DiffusionBlock(img_size=img_size, in_channels=3, kernel=1)
        e = time.time()
        print(f"model initialized in {(e-s):.2f} seconds ")
        input_tensor = torch.randn(4, 3, *img_size)

        # Run the forward pass and check the output
        output = model(input_tensor)
        self.assertEqual(output.shape, (4, 3, 32, 32)) # output should have shape (batch_size, num_classes)

        # Testing completed
        print("Testing Diffusion Block Forward COMPLETED!!!")
