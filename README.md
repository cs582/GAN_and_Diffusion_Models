# Product Name
> From scratch implementation of GANs and Diffusion deep learning models using PyTorch.

![Python version][python-image]
![PyTorch version][pytorch-image]
![Scikit Learn version][scikit-learn-image]
![Matplotlib version][matplotlib-image]
![Pandas version][pandas-image]
![NumPy version][numpy-image]

This project implements the ground breaking papers on Generative Adversarial Networks and Diffusion Networks.

Generative Adversarial Networks
> Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2020). Generative adversarial networks. Communications of the ACM, 63(11), 139-144.

Diffusion Networks
> Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015, June). Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning (pp. 2256-2265). PMLR.

![](header.png)

## Installation

OS X & Linux:

```sh
pip install -r requirements
```

## Unit tests

Unit tests check the backbones and building blocks of the neural networks.

```sh
python test.py
```


## Usage example

These are the arguments taken by the main.py script.

```sh
-h, --help            show this help message and exit
-epoch EPOCH          Number of epochs.
-timesteps TIMESTEPS  Number of timesteps.
-batch_size BATCH_SIZE
Batch size.
-lr LR                Learning Rate.
-latent_vs LATENT_VS  Latent vector size.
-model MODEL          Choose model to train.
-dataset DATASET      Choose dataset.
```

To train the model you can run.

```sh
python main.py
```


## About the Author

Carlos Gustavo Salas Flores – [LinkedIn](https://www.linkedin.com/in/carlosgustavosalas/) – yuseicarlos2560@gmail.com

Distributed under the MIT license. See ``LICENSE.txt`` for more information.

[https://github.com/cs582](https://github.com/cs582/)


<!-- Markdown link & img dfn's -->
[python-image]: https://img.shields.io/badge/Python-3.8.5-blue?style=flat-square]
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.9.0-orange?style=flat-square]
[scikit-learn-image]: https://img.shields.io/badge/scikit--learn-0.24.1-blue?style=flat-square]
[matplotlib-image]: https://img.shields.io/badge/Matplotlib-3.3.4-orange?style=flat-square]
[pandas-image]: https://img.shields.io/badge/Pandas-1.2.3-blue?style=flat-square]
[numpy-image]: https://img.shields.io/badge/NumPy-1.20.1-orange?style=flat-square]
