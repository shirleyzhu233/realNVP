"""Training procedure for real NVP.
"""

import argparse

import torch, torchvision
import torch.distributions as distributions
import torch.optim as optim
import numpy as np

import realnvp, data_utils

class Hyperparameters():
    def __init__(self, base_dim, res_blocks, bottleneck, 
        skip, weight_norm, coupling_bn, affine):
        """Instantiates a set of hyperparameters used for constructing layers.

        Args:
            base_dim: features in residual blocks of first few layers.
            res_blocks: number of residual blocks to use.
            bottleneck: True if use bottleneck, False otherwise.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if batchnorm coupling layer output, False otherwise.
            affine: True if use affine coupling, False if use additive coupling.
        """
        self.base_dim = base_dim
        self.res_blocks = res_blocks
        self.bottleneck = bottleneck
        self.skip = skip
        self.weight_norm = weight_norm
        self.coupling_bn = coupling_bn
        self.affine = affine

def main(args):
    device = torch.device("cuda:0")

    # model hyperparameters
    dataset = args.dataset
    batch_size = args.batch_size
    hps = Hyperparameters(
        base_dim = args.base_dim, 
        res_blocks = args.res_blocks, 
        bottleneck = args.bottleneck, 
        skip = args.skip, 
        weight_norm = args.weight_norm, 
        coupling_bn = args.coupling_bn, 
        affine = args.affine)
    scale_reg = 5e-5    # L2 regularization strength

    # optimization hyperparameters
    lr = args.lr
    momentum = args.momentum
    decay = args.decay

    # prefix for images and checkpoints
    filename = 'bs%d_' % batch_size \
             + 'normal_' \
             + 'bd%d_' % hps.base_dim \
             + 'rb%d_' % hps.res_blocks \
             + 'bn%d_' % hps.bottleneck \
             + 'sk%d_' % hps.skip \
             + 'wn%d_' % hps.weight_norm \
             + 'cb%d_' % hps.coupling_bn \
             + 'af%d' % hps.affine \

    # load dataset
    trainset, datainfo = data_utils.load(dataset)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=batch_size, shuffle=True, num_workers=2)

    prior = distributions.Normal(   # isotropic standard normal distribution
        torch.tensor(0.).to(device), torch.tensor(1.).to(device))
    flow = realnvp.RealNVP(datainfo=datainfo, prior=prior, hps=hps).to(device)
    optimizer = optim.Adam(flow.parameters(), lr=lr, betas=(momentum, decay))
    total_iter = 0

    # load model checkpoint
    try:
        path = 'models/' + dataset + '/' + filename + '.tar'
        ckpt = torch.load(path)
        flow.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        total_iter = ckpt['total_iter']
        print('Load checkpoint: success!')
        print('\tstart from iter: %d loss: %.3f' % (total_iter, ckpt['loss']))
    except:
        print('No checkpoint found. Train from scratch.')

    train = True
    running_loss = 0
    image_size = datainfo.channel * datainfo.size**2    # full image dimension

    while train:
        for _, data in enumerate(trainloader, 1):
            flow.train()
            if total_iter == args.max_iter:
                train = False
                break

            total_iter += 1
            optimizer.zero_grad()

            inputs, _ = data
            # log-determinant of Jacobian from the logit transform
            inputs, log_det_J = data_utils.logit_transform(inputs)
            inputs = inputs.to(device)
            log_det_J = log_det_J.to(device)

            # log-likelihood of input minibatch
            log_ll, weight_scale = flow(inputs)
            log_ll = (log_ll + log_det_J).mean()

            # add L2 regularization on scaling factors
            loss = -log_ll + scale_reg * weight_scale
            running_loss += float(loss)

            loss.backward()
            optimizer.step()

            if total_iter % 500 == 0:
                mean_loss = running_loss / 500
                bit_per_dim = (float(-log_ll) + np.log(256.) * image_size) \
                            / (image_size * np.log(2.))
                print('iter %s:' % total_iter, 
                      'loss = %.3f' % mean_loss, 
                      'bits/dim = %.3f' % bit_per_dim)
                running_loss = 0.

                flow.eval()
                with torch.no_grad():
                    z, _ = flow.f(inputs)
                    reconst = flow.g(z)
                    reconst, _ = data_utils.logit_transform(reconst, reverse=True)
                    samples = flow.sample(args.sample_size)
                    samples, _ = data_utils.logit_transform(samples, reverse=True)
                    torchvision.utils.save_image(utils.make_grid(reconst),
                        './reconstruction/' + dataset + '/' + filename + '_%d.png' % total_iter)
                    torchvision.utils.save_image(utils.make_grid(samples),
                        './samples/' + dataset + '/' + filename + '_%d.png' % total_iter)

                if total_iter % 20000 == 0:
                    torch.save({
                        'total_iter': total_iter,
                        'loss': mean_loss, 
                        'model_state_dict': flow.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'batch_size': batch_size,
                        'base_dim': hps.base_dim,
                        'res_blocks': hps.res_blocks,
                        'bottleneck': hps.bottleneck,
                        'skip': hps.skip,
                        'weight_norm': hps.weight_norm,
                        'coupling_bn': hps.coupling_bn,
                        'affine': hps.affine}, 
                        './models/' + dataset + '/' + filename + '.tar')
                    print('Checkpoint saved.')

    print('Training finished.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CIFAR-10 realNVP PyTorch implementation')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='cifar10')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=64)
    parser.add_argument('--base_dim',
                        help='features in residual blocks of first few layers.',
                        type=int,
                        default=64)
    parser.add_argument('--res_blocks',
                        help='number of residual blocks per group.',
                        type=int,
                        default=8)
    parser.add_argument('--bottleneck',
                        help='whether to use bottleneck in residual blocks.',
                        type=int,
                        default=0)
    parser.add_argument('--skip',
                        help='whether to use skip connection in coupling layers.',
                        type=int,
                        default=1)
    parser.add_argument('--weight_norm',
                        help='whether to apply weight normalization.',
                        type=int,
                        default=1)
    parser.add_argument('--coupling_bn',
                        help='whether to apply batchnorm after coupling layers.',
                        type=int,
                        default=1)
    parser.add_argument('--affine',
                        help='whether to use affine coupling.',
                        type=int,
                        default=1)
    parser.add_argument('--max_iter',
                        help='maximum number of iterations.',
                        type=int,
                        default=100000)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)
    parser.add_argument('--momentum',
                        help='beta1 in Adam optimizer.',
                        type=float,
                        default=0.9)
    parser.add_argument('--decay',
                        help='beta2 in Adam optimizer.',
                        type=float,
                        default=0.999)
    args = parser.parse_args()
    main(args)