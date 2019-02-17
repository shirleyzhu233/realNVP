"""Training procedure for real NVP.
"""

import argparse

import torch, torchvision
import torch.distributions as distributions
import torch.optim as optim
import torchvision.utils as utils

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
    train_split, val_split, data_info = data_utils.load(dataset)
    train_loader = torch.utils.data.DataLoader(train_split,
        batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_split,
        batch_size=batch_size, shuffle=False, num_workers=2)

    prior = distributions.Normal(   # isotropic standard normal distribution
        torch.tensor(0.).to(device), torch.tensor(1.).to(device))
    flow = realnvp.RealNVP(datainfo=data_info, prior=prior, hps=hps).to(device)
    optimizer = optim.Adamax(flow.parameters(), lr=lr, betas=(momentum, decay), eps=1e-7)
    
    epoch = 0
    running_loss = 0.
    running_log_ll = 0.
    optimal_log_ll = float('-inf')
    early_stop = 0

    image_size = data_info.channel * data_info.size**2    # full image dimension

    while epoch < args.max_epoch:
        epoch += 1
        print('Epoch %d:' % epoch)
        flow.train()
        for batch_idx, data in enumerate(train_loader, 1):
            optimizer.zero_grad()
            x, _ = data
            # log-determinant of Jacobian from the logit transform
            x, log_det = data_utils.logit_transform(x)
            x = x.to(device)
            log_det = log_det.to(device)

            # log-likelihood of input minibatch
            log_ll, weight_scale = flow(x)
            log_ll = (log_ll + log_det).mean()

            # add L2 regularization on scaling factors
            loss = -log_ll + scale_reg * weight_scale
            running_loss += loss.item()
            running_log_ll += log_ll.item()

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                bit_per_dim = (-log_ll.item() + np.log(256.) * image_size) \
                    / (image_size * np.log(2.))
                print('[%d/%d]\tloss: %.3f\tlog-ll: %.3f\tbits/dim: %.3f' % \
                    (batch_idx*batch_size, len(train_loader.dataset), 
                        loss.item(), log_ll.item(), bit_per_dim))
        
        mean_loss = running_loss / batch_idx
        mean_log_ll = running_log_ll / batch_idx
        mean_bit_per_dim = (-mean_log_ll + np.log(256.) * image_size) \
             / (image_size * np.log(2.))
        print('===> Average train loss: %.3f' % mean_loss)
        print('===> Average train log-likelihood: %.3f' % mean_log_ll)
        print('===> Average train bit_per_dim: %.3f' % mean_bit_per_dim)
        running_loss = 0.
        running_log_ll = 0.

        flow.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader, 1):
                x, _ = data
                x, log_det = data_utils.logit_transform(x)
                x = x.to(device)
                log_det = log_det.to(device)

                # log-likelihood of input minibatch
                log_ll, weight_scale = flow(x)
                log_ll = (log_ll + log_det).mean()

                # add L2 regularization on scaling factors
                loss = -log_ll + scale_reg * weight_scale
                running_loss += loss.item()
                running_log_ll += log_ll.item()

            mean_loss = running_loss / batch_idx
            mean_log_ll = running_log_ll / batch_idx
            mean_bit_per_dim = (-mean_log_ll + np.log(256.) * image_size) \
                / (image_size * np.log(2.))
            print('===> Average validation loss: %.3f' % mean_loss)
            print('===> Average validation log-likelihood: %.3f' % mean_log_ll)
            print('===> Average validation bits/dim: %.3f' % mean_bit_per_dim)
            running_loss = 0.
            running_log_ll = 0.

            samples = flow.sample(args.sample_size)
            samples, _ = data_utils.logit_transform(samples, reverse=True)
            utils.save_image(utils.make_grid(samples),
                './samples/' + dataset + '/' + filename + '_ep%d.png' % epoch)

        if mean_log_ll > optimal_log_ll:
            early_stop = 0
            optimal_log_ll = mean_log_ll
            torch.save(flow, './models/' + dataset + '/' + filename + '.model')
            print('[MODEL SAVED]')
        else:
            early_stop += 1
            if early_stop >= 100:
                break
        
        print('--> Early stopping %d/100 (BEST validation log-likelihood: %.3f)' \
            % (early_stop, optimal_log_ll))

    print('Training finished at epoch %d.' % epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Real NVP PyTorch implementation')
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
    parser.add_argument('--max_epoch',
                        help='maximum number of training epoches.',
                        type=int,
                        default=500)
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