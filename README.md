# Auto-Encoding-Variational-Bayes (VAE)

<img src="images/vae_image.png" >

This project implements the models proposed in [ Auto-Encoding Variational Bayes (VAE) paper](https://arxiv.org/abs/1312.6114)
written by Diederik P Kingma and Max Welling.

The models were ran over the following datasets:
- MNIST
- MNIST Fashion

The results of the models can be found in [results.pdf](./results.pdf).

Below are samples generated by the trained VAE model on MNIST and Fashion-MNIST datasets:

- Fashion-MNIST Samples: Images generated after training a VAE on the Fashion-MNIST dataset. The model has learned to capture and reproduce the structure of clothing items like shoes, shirts, and bags.

  <img src="samples/fashion-mnist_batch128_mid100_.png">

- MNIST Samples: Digits generated by a VAE trained on the MNIST dataset.

  <img src="samples/mnist_batch128_mid100_.png">


Here’s a `README.md`-style section in the format you asked for, tailored to the **VAE training project**:

---

## Setup

### Prerequisites
- Python 3.x
- NumPy
- torch
- torchvision
- matplotlib

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Running the Code

To train the VAE model on either MNIST or Fashion-MNIST:

```bash
python main.py
```

You can customize the training with the following command-line arguments:

| Argument         | Description                                           | Default         |
|------------------|-------------------------------------------------------|-----------------|
| `--dataset`      | Dataset to train on: `mnist` or `fashion-mnist`      | `mnist`         |
| `--batch_size`   | Number of images per mini-batch                      | `128`           |
| `--epochs`       | Number of training epochs                            | `50`            |
| `--sample_size`  | Number of images to generate during sampling         | `64`            |
| `--latent-dim`   | Dimensionality of the latent space (z vector size)   | `100`           |
| `--lr`           | Initial learning rate for the optimizer              | `1e-3`          |

#### Example with Custom Parameters:

```bash
python main.py --dataset fashion-mnist --batch_size 64 --epochs 40 --latent-dim 50 --lr 0.0005
```

---

### Output Files

- **Generated Samples**: After training, 100 samples are generated and saved in the `./samples/` folder as:
  - `fashion-mnist_batch128_mid100_.png`
  - `mnist_batch128_mid100_.png`

- **Loss Tracking**: The training and testing loss for each epoch is saved in a pickle file:
  - `loss_batches_mnist.pkl` or `loss_batches_fashion-mnist.pkl`