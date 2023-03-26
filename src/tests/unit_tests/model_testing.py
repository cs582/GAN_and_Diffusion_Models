import unittest
import torch
from src.models.classifiers.CNNs import MicroCNN, MiniCNN
from src.models.building_blocks import SimpleCNNBlock, MLP
from src.models.GAN_models import MNISTGenerator


class TestBuildingBlocks(unittest.TestCase):
    def test_simplecnnblock_forward(self):
        # Print to console
        print("Testing SimpleCNNBlock Forward...")

        # Initialize the model and input tensor
        model = SimpleCNNBlock(in_channels=3, kernel_size=3, padding=0, stride=1, out_channels=6)
        input_tensor = torch.randn(1, 3, 28, 28)

        # Run the forward pass and check the output
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 6, 26, 26)) # output should have shape (batch_size, num_classes)

        # Testing completed
        print("Testing SimpleCNNBlock COMPLETED!!!")

    def test_mlp_forward(self):
        # Print to console
        print("Testing MLP Forward...")

        # Initialize the model and input tensor
        model = MLP(in_size=512, out_size=10)
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
        model = MLP(in_size=512, out_size=10)
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
        model = MicroCNN(in_channels=1, out_size=10)
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
        model = MicroCNN(in_channels=1, out_size=10)
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
        model = MiniCNN(in_channels=3, out_size=10)
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
        model = MiniCNN(in_channels=3, out_size=10)
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
        model = MNISTGenerator(in_size=10, out_shape=(28,28))
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
        model = MNISTGenerator(in_size=10, out_shape=(28,28))
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
