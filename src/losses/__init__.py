from losses.user_losses.image_reg_loss import ImageRegLoss
from resources.const import REGULARIZATION_TYPES
import logging
import os.path as osp
import mmcv
import importlib


logger = logging.getLogger('base')


def parse_loss_str(args):
    loss_folder = osp.join(osp.dirname(osp.abspath(__file__)), 'image_losses')

    loss_filenames = [
        osp.splitext(osp.basename(v))[0] for v in mmcv.scandir(loss_folder)
        if v.endswith('_loss.py')
    ]

    loss_modules = [
        importlib.import_module(f'losses.image_losses.{loss_filename}')
        for loss_filename in loss_filenames
    ]

    losses, regs = {}, {}
    logger.info(f'Losses: {args.loss_str}')

    for hyper_param, loss_type in (x.split('*') for x in args.loss_str.split('+')):
        if loss_type.endswith('Loss'):
            found = False
            for loss_module in loss_modules:
                loss_cls = getattr(loss_module, loss_type, None)
                if loss_cls is not None:
                    losses[loss_type] = (loss_cls(args), float(hyper_param))
                    found = True
                    break
            if not found:
                raise NotImplementedError(f'Unrecognized loss {loss_type}!')
        elif loss_type.endswith('Reg'):
            if loss_type in REGULARIZATION_TYPES:
                regs[loss_type] = float(hyper_param)
            else:
                raise NotImplementedError(f'Unrecognized regularization {loss_type}!')
        else:
            raise ValueError(f'{loss_type} is neither loss nor regularization')

    logger.info(f'Loss: {losses}')
    logger.info(f'Loss: {regs}')

    return losses, regs


def define_loss(args):
    if args.model == 'ImageRegModel':
        return ImageRegLoss(*parse_loss_str(args))
    else:
        raise NotImplementedError(f'Unregconized model {args.model}!')
