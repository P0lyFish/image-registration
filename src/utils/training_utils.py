import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import numpy as np
import argparse
from datetime import datetime
import logging
import os

from resources.const import IMG_NORMALIZE_MEAN, IMG_NORMALIZE_STD


def define_optimizer(my_model, args):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'ADAMax':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': 1e-08}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    logger = logging.getLogger('base')
    logger.info('Optimizer function: {args.optimizer}')
    logger.info('Learning rate: {args.lr}')
    logger.info('Weight decay: {args.weights_decay}')

    return optimizer_function(trainable, **kwargs)


def define_scheduler(my_optimizer, args):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    logger = logging.getLogger('base')
    logger.info(f'Decay type: {args.decay_type}')

    return scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Compression-Driven Frame Interpolation Training')

    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)

    # Directory Setting
    parser.add_argument('--dataset', type=str, required=True)

    # Learning Options
    parser.add_argument('--num_epochs', type=int, default=100, help='Max Epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--loss_str', type=str, default='1*CharbonnierLoss+0.8*GSpatialReg', help='loss function configuration')
    parser.add_argument('--img_log_freq', type=int, default=500, help='saving image frequency')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=int, default=10000, help='learning rate decay per N epochs')
    parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='ADAMax', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'), help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--load_epoch', type=int, default=-1, help='Load checkpoint from a specific epoch')
    parser.add_argument('--use_parallel', type=bool, default=True, help='Use parallel training')
    parser.add_argument('--val_metrics', type=list, default=['psnrMetric'], help='validation metrics')

    # Options for AdaCoF
    parser.add_argument('--kernel_size', type=int, default=11)
    parser.add_argument('--dilation', type=int, default=2)

    args = parser.parse_args()

    return args


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def moduleNormalize(frame):
    return torch.cat([(frame[:, 0:1, :, :] - 0.4631), (frame[:, 1:2, :, :] - 0.4352), (frame[:, 2:3, :, :] - 0.3990)], 1)


def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data ** 2 + epsilon ** 2))


def unnormalize(img, max_pixel_value=255.):
    # img: HxWxC
    mean, std = np.array(IMG_NORMALIZE_MEAN), np.array(IMG_NORMALIZE_STD)

    return img * (std * max_pixel_value) + mean * max_pixel_value
