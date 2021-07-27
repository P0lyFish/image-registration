from data.vimeo90k_dataset import Vimeo90kDataset
import torch
import logging


def define_dataloaders(args):
    if args.dataset == 'Vimeo90k':
        train_dataset = Vimeo90kDataset('train')
        val_dataset = Vimeo90kDataset('val')
    else:
        raise NotImplementedError(f'Unrecognized {args.dataset} dataset!')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )

    logger = logging.getLogger('base')
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'Batch size: {args.batch_size}')
    logger.info(f'Number of training data points: {len(train_dataset)}')
    logger.info(f'Number of validation data points: {len(val_dataset)}')

    return train_dataloader, val_dataloader
