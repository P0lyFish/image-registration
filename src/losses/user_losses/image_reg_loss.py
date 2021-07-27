import torch.nn as nn


class ImageRegLoss(nn.Module):
    def __init__(self, losses, reg_hyper_params):
        super(ImageRegLoss, self).__init__()

        self.losses = losses
        self.reg_hyper_params = reg_hyper_params

    def forward(self, gt, pred, regs):
        ret = {}

        total_loss = 0.
        for loss_type, (loss_func, hyper_param) in self.losses.items():
            loss = loss_func(gt, pred) * hyper_param
            total_loss = total_loss + loss

            ret[loss_type] = loss

        for reg_type, hyper_param in self.reg_hyper_params.items():
            loss = regs[reg_type].mean() * hyper_param
            total_loss = total_loss + loss

            ret[reg_type] = loss

        ret['totalLoss'] = total_loss

        return ret
