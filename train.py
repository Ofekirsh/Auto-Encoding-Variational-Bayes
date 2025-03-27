"""Training procedure for NICE.
"""

import argparse
import pickle

import torch, torchvision
from torchvision import transforms
import numpy as np
from VAE import Model

def train(vae, trainloader, optimizer, epoch):
    vae.train()  # set to training mode
    #TODO
    loss_epoch = 0

    for x, y in trainloader:
        x = x.to(vae.device)
        optimizer.zero_grad()
        recon, mu_enc, logvar_enc = vae(x)
        loss = vae.loss(x, recon, mu_enc, logvar_enc)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    loss = loss_epoch / len(trainloader)
    print(f"loss train {epoch} - {loss}")
    return loss

def test(vae, testloader, filename, epoch=None):
    vae.eval()  # set to inference mode
    with torch.no_grad():
        #TODO
        if epoch == None:
            samples = vae.sample(100)
            torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                       './samples/' + filename + '.png')

        loss_epoch = 0
        for x, y in testloader:
            x = x.to(vae.device)
            recon, mu_enc, logvar_enc = vae(x)
            loss = vae.loss(x, recon, mu_enc, logvar_enc)
            loss_epoch += loss.item()
        loss = loss_epoch / len(testloader)
        print(f"loss test {epoch} - {loss}")
        return loss


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]

    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    vae = Model(latent_dim=args.latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=args.lr)

    #TODO
    loss_batches_train = []
    loss_batches_test = []
    for epoch in range(args.epochs):
        loss_train_batch = train(vae, trainloader, optimizer, epoch)
        loss_test_batch = test(vae, testloader, filename, epoch)

        loss_batches_train.append(loss_train_batch)
        loss_batches_test.append(loss_test_batch)

    test(vae, testloader, filename, epoch=None)

    # Filepath to save the pickle file
    file_path = f'loss_batches_{args.dataset}.pkl'

    # Save the lists to a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump({'loss_batches_train': loss_batches_train, 'loss_batches_test': loss_batches_test}, f)

    print(f"Data saved to {file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
