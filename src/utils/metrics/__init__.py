import importlib
import mmcv
import os.path as osp
import logging


def define_metrics(args):
    metric_folder = osp.dirname(osp.abspath(__file__))

    metric_filenames = [
        osp.splitext(osp.basename(v))[0] for v in mmcv.scandir(metric_folder)
        if v.endswith('_metric.py')
    ]

    metric_modules = [
        importlib.import_module(f'utils.metrics.{metric_filename}')
        for metric_filename in metric_filenames
    ]

    metrics = {}
    for metric_name in args.val_metrics:
        for metric_module in metric_modules:
            metric_cls = getattr(metric_module, metric_name, None)
            if metric_cls is not None:
                found = True
                metrics[metric_name] = metric_cls
                break

        if not found:
            raise NotImplementedError(f'Unrecognized metric {metric_name}')

    logger = logging.getLogger('base')
    logger.info(f'Validation metric(s): {metrics.keys()}')
    return metrics
