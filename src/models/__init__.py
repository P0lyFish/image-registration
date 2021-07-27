from models.image_reg_model import ImageRegModel
import logging


def define_model(args):
    logger = logging.getLogger('base')

    if args.model == 'ImageRegModel':
        model = ImageRegModel(args)
    else:
        raise NotImplementedError(f'Unregconized model {args.model}!')

    logger.info(model)
    return model
