import torch
from utils.training_utils import unnormalize


def psnrMetric(gt, pred):
    assert gt.shape == pred.shape, (
        f'Image shapes are differnet: {gt.shape}, {pred.shape}.'
    )

    gt = gt.permute(0, 2, 3, 1).cpu().detach().squeeze()
    pred = pred.permute(0, 2, 3, 1).cpu().detach().squeeze()
    gt, pred = unnormalize(gt), unnormalize(pred)

    mse = torch.mean((gt - pred)**2)
    if mse == 0:
        return float('inf')
    return (20. * torch.log10(255. / torch.sqrt(mse))).item()
