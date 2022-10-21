import torch
import numpy as np
import pandas as pd
from typing import Callable, NoReturn, Optional, Tuple
from tqdm import tqdm
import copy
from torch.utils.data import Dataset, DataLoader
from ..utils.utils import SwapNoiseMasker, EarlyStopping, AverageMeter
from .base_model import BaseDLModel

class DeepStackMLP(torch.nn.Module):
    def __init__(self,
                 len_cat,
                 len_num,
                 dropout_ratio,
                 hidden_size,
                 lower_bound=0,
                 upper_bound=10.5):
        super().__init__()

        self.len_total = len_cat + len_num
        self.hidden_size = hidden_size
        self.half_hidden_size = int(self.hidden_size / 2)
        self.dropout_ratio = dropout_ratio
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.len_total, out_features=self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.Dropout(self.dropout_ratio)
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hidden_size, out_features=self.half_hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.half_hidden_size),
            torch.nn.Dropout(self.dropout_ratio)
        )
        self.layer_3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.half_hidden_size, out_features=self.half_hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.half_hidden_size),
            torch.nn.Dropout(self.dropout_ratio)
        )
        self.last_linear = torch.nn.Linear(self.half_hidden_size, 1)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def net(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.last_linear(x)

    def forward(self, x):
        return torch.sigmoid(self.net(x) * (self.upper_bound - self.lower_bound) + self.lower_bound)

    def loss(self, x, y):
        mlp_loss = torch.nn.functional.binary_cross_entropy(x, y)
        return mlp_loss

class MLP(BaseDLModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

    def _train(self,
               train_dl: DataLoader,
               valid_dl: DataLoader,
               len_cat: int,
               len_num: int) -> DeepStackMLP:

        model = DeepStackMLP(
            len_cat=len_cat,
            len_num=len_num,
            dropout_ratio=self.config.model.params.dropout_ratio,
            hidden_size=self.config.model.params.hidden_size
        ).cuda()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.model.optimizer.init_lr,
            weight_decay=self.config.model.optimizer.l2_reg
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=1 / 3,
                                                               patience=self.config.model.scheduler.patience,
                                                               verbose=self.config.model.scheduler.verbose,
                                                               cooldown=self.config.model.scheduler.cooldown,
                                                               min_lr=self.config.model.scheduler.min_lr)

        earlystopper = EarlyStopping(mode='min',
                                     min_delta=self.config.model.earlystopper.min_delta,
                                     patience=self.config.model.earlystopper.patience,
                                     percentage=False,
                                     verbose=self.config.model.earlystopper.verbose)
        best_score = float('inf')

        for epoch in tqdm(range(self.config.model.iterations)):
            model.train()
            meter = AverageMeter()
            # train
            for i, (x, target) in enumerate(train_dl):
                x, target = x.cuda(), target.cuda()
                optimizer.zero_grad()
                loss = model.loss(model.forward(x), target.unsqueeze(1))
                loss.backward()
                optimizer.step()
                meter.update(loss.detach().cpu().numpy())
            train_loss = meter.overall_avg
            # valid
            meter.reset()
            with torch.no_grad():
                for i, (x, target) in enumerate(valid_dl):
                    x, target = x.cuda(), target.cuda()
                    loss = model.loss(model.forward(x), target.unsqueeze(1))
                    meter.update(loss.detach().cpu().numpy())
            valid_loss = meter.overall_avg

            if epoch % self.config.model.eval_verbose == 0:
                print('\repoch {:4d} - train loss {:6.4f} - valid loss {:6.4f}'.format(epoch, train_loss, valid_loss))

            scheduler.step(valid_loss)
            # checkpointing
            if valid_loss < best_score:
                best_score = valid_loss
                best_model = copy.deepcopy(model)
            # early stopping
            if earlystopper.step(valid_loss):
                break

        return best_model