from src.tests.MNIST_tests.GAN_MNIST import run
import argparse

parser = argparse.ArgumentParser(
    prog='GAN and Diffusion Models',
    description='Training GAN or Diffusion models replicating the original papers'
)

parser.add_argument('-epochs', type=int, default=10, help="Number of epochs.")
parser.add_argument('-bsize', type=int, default=64, help="Batch size.")
parser.add_argument('-lr', type=float, default=1e-4, help="Learning Rate.")
parser.add_argument('-vector_size', type=int, default=100, help="Latent vector size.")

args = parser.parse_args()

if __name__ == "__main__":
    epochs = args.epochs
    batch_size = args.bsize
    lr = args.lr
    latent_vector_size = args.vector_size

    run(epochs, batch_size, lr, latent_vector_size)

