from tqdm import tqdm
import os.path as osp
import os
import logging
import torch
from resources.const import TRAINING_STATE_SAVE_PATH_FORMAT
from resources.const import MODEL_SAVE_PATH_FORMAT
from resources.const import CHECKPOINT_PATH
from torch.utils.tensorboard import SummaryWriter
import utils.training_utils as training_utils

from models import define_model
from losses import define_loss
from utils.metrics import define_metrics
from data import define_dataloaders


class SupervisedTrainer:
    def __init__(self, args):
        self.exp_name = args.exp_name
        self.device = torch.device("cuda")
        self.initialize_training_folders(args.load_epoch == -1)

        self.model = define_model(args).to(self.device)

        self.loss_func = define_loss(args).to(self.device)

        self.train_dataloader, self.val_dataloader = define_dataloaders(args)
        self.optimizer = training_utils.define_optimizer(self.model, args)
        self.scheduler = training_utils.define_scheduler(self.optimizer, args)

        self.num_epochs = args.num_epochs

        self.metric_funcs = define_metrics(args)

        if args.use_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.load(args.load_epoch)

    def initialize_training_folders(self, from_scratch):
        exp_path = osp.join(CHECKPOINT_PATH, self.exp_name)

        if from_scratch and osp.isdir(exp_path):
            timestamp = training_utils.get_timestamp()
            os.rename(exp_path, osp.join(osp.dirname(exp_path), self.exp_name + '_' + timestamp))

        exp_path = osp.join(CHECKPOINT_PATH, self.exp_name)

        if from_scratch:
            os.makedirs(exp_path)
            os.makedirs(osp.join(exp_path, 'models'), exist_ok=True)
            os.makedirs(osp.join(exp_path, 'training_states'), exist_ok=True)

        self.writer = SummaryWriter(log_dir=exp_path)

        training_utils.setup_logger('base', exp_path, screen=True, tofile=True)
        self.logger = logging.getLogger('base')

    def train(self):
        while self.current_epoch < self.num_epochs:
            self.train_one_epoch()
            self.validation()
            self.save()

    def train_one_epoch(self):
        self.model.train()

        progressBar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f'Running epoch {self.current_epoch}/{self.num_epochs}'
        )

        for batch_idx, data_point in progressBar:
            self.optimizer.zero_grad()

            out = self.model(data_point['input'])
            pred = out['pred']
            regs = out['regs']
            losses = self.loss_func(data_point['gt'].to(self.device), pred, regs)

            losses['totalLoss'].backward()
            self.optimizer.step()
            self.scheduler.step()
            self.current_iter += 1

            self.log_losses(losses)

        self.log_imgs({'pred': pred})
        self.log_imgs(data_point)

    def log_losses(self, losses):
        for loss_type, loss in losses.items():
            if loss_type.endswith('Loss'):
                self.writer.add_scalar(f'Losses/{loss_type}', loss.item(), self.current_iter)
            elif loss_type.endswith('Reg'):
                self.writer.add_scalar(f'Regularizations/{loss_type}', loss.item(), self.current_iter)
            else:
                raise ValueError(f'Unrecognized loss {loss_type}!')

    def log_imgs(self, data_point, current_tag=None):
        if type(data_point) is torch.Tensor:
            B, C, H, W = data_point.shape
            img = data_point[0].cpu().detach().numpy().transpose((1, 2, 0))
            img = training_utils.unnormalize(img) / 255.
            self.writer.add_image(current_tag, img, self.current_iter, dataformats='HWC')
            return

        if type(data_point) is dict:
            for k, v in data_point.items():
                new_tag = k if current_tag is None else current_tag + '/' + k
                self.log_imgs(v, new_tag)

    def save(self):
        """
        Save training state during training,
        which will be used for resuming
        """
        self.logger.info('Saving epoch #{self.current_epoch}...')

        state = {
            "epoch": self.current_epoch,
            "scheduler": self.scheduler,
            "optimizer": self.optimizer,
            'current_iter': self.current_iter
        }

        training_state_save_path = \
            TRAINING_STATE_SAVE_PATH_FORMAT.format(self.exp_name, self.current_epoch)
        torch.save(state, training_state_save_path)

        model_save_path = MODEL_SAVE_PATH_FORMAT.format(self.exp_name, self.current_epoch)
        torch.save(self.model.state_dict(), model_save_path)

    def load(self, load_epoch):
        if load_epoch == -1:
            self.logger.info('Training model from scratch...')
            self.current_epoch = 0
            self.current_iter = 0
            return

        self.logger.info(f'Loading pretrained model from epoch #{load_epoch}')

        state_path = TRAINING_STATE_SAVE_PATH_FORMAT.format(self.exp_name, load_epoch)
        model_path = MODEL_SAVE_PATH_FORMAT.format(self.exp_name, load_epoch)

        if not osp.isfile(state_path):
            raise ValueError(f'Training state for epoch #{load_epoch} not found!')
        if not osp.isfile(model_path):
            raise ValueError(f'Model weights for epoch #{load_epoch} not found!')

        state = torch.load()

        self.optimizer = state['optimizer']
        self.scheduler = state['scheduler']
        self.current_epoch = state['epoch']
        self.current_iter = state['current_iter']

        self.model.load_state_dict(torch.load(model_path))

        self.current_epoch = load_epoch + 1

    def validation(self):
        if self.val_dataloader is None:
            self.logger.warning('No validation dataloader was given. Skipping validation...')
            return

        progressBar = tqdm(
            enumerate(self.val_dataloader),
            total=len(self.val_dataloader),
            desc=f'Validating #{self.current_epoch}/{self.num_epochs}'
        )

        self.model.eval()

        avg = {}
        for metric_name in self.metric_funcs.keys():
            avg[metric_name] = 0.

        with torch.no_grad():
            for batch_idx, dataPoint in progressBar:
                out = self.model(dataPoint['input'])
                for metric_name, metric_func in self.metric_funcs.items():
                    metric = metric_func(out['pred'], dataPoint['gt'])
                    avg[metric_name] += metric

        for metric_name, score in avg.items():
            score /= len(self.val_dataloader)
            self.logger.info(f"Average {metric_name}: {score:.4f}")
            self.writer.add_scalar(f"Val/{metric_name}", score, self.current_epoch)


def main():
    args = training_utils.parse_args()

    trainer = SupervisedTrainer(args)

    trainer.train()


if __name__ == '__main__':
    main()
