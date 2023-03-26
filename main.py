from src.tests import MNIST_tests
import argparse

parser = argparse.ArgumentParser(
    prog='GAN and Diffusion Models',
    description='Training GAN or Diffusion models replicating the original papers'
)

parser.add_argument('-epochs', type=int, default=10, help="Number of epochs.")
parser.add_argument('-bsize', type=int, default=64, help="Batch size.")
parser.add_argument('-lr', type=float, default=1e-4, help="Learning Rate.")
parser.add_argument('-vector_size', type=int, default=100, help="Latent vector size.")
parser.add_argument('-model', type=str, default="GAN", help="Choose model to train.")
parser.add_argument('-dataset', type=str, default="MNIST", help="Choose dataset.")


args = parser.parse_args()


if __name__ == "__main__":

    if args.model == "GAN":
        if args.dataset == "MNIST":
            MNIST_tests.GAN_MNIST.run(epochs=args.epochs, batch_size=args.batch_size, latent_vector_size=args.vector_size, lr=args.lr)

    if args.model == "Diffusion":
        if args.dataset == "MNIST":
            MNIST_tests.DIFF_MNIST.run(batch_size=args.batch_size, latent_vector_size=args.vector_size, lr=args.lr)



