import torch
import numpy as np
import pandas as pd
from typing import Callable, NoReturn, Optional, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from ..dataset.dataset import get_dae_dataset, get_dataset_with_cat_handler
from ..utils.utils import SwapNoiseMasker, EarlyStopping, AverageMeter
from .base_boosting import BaseModel, ModelResult

class DeepStackDAE(torch.nn.Module):
    def __init__(self,
                 num_cats,
                 num_conts,
                 hidden_size,
                 emphasis):
        super().__init__()

        post_encoding_input_size = num_cats + num_conts
        self.hidden_size = hidden_size
        self.linear_1 = torch.nn.Linear(in_features=post_encoding_input_size, out_features=self.hidden_size)
        self.linear_2 = torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.linear_3 = torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.linear_4 = torch.nn.Linear(in_features=self.hidden_size, out_features=post_encoding_input_size)

        self.num_cats = num_cats
        self.num_conts = num_conts
        self.emphasis = emphasis
        self.upper_bound = 10.5
        self.lower_bound = 0

    def forward(self, x):
        act_1 = torch.nn.functional.relu(self.linear_1(x))
        act_2 = torch.nn.functional.relu(self.linear_2(act_1))
        act_3 = torch.nn.functional.relu(self.linear_3(act_2))
        out = self.linear_4(act_3)
        return act_1, act_2, act_3, out

    def feature(self, x):
        return torch.cat(self.forward(x)[:-1], dim=1)

    def split(self, t):
        return torch.split(t, [self.num_cats, self.num_conts], dim=1)

    def loss(self, x, y, mask=None, weights=[1, 1]):
        if mask is None:
            mask = torch.ones(x.shape).to(x.device)

        x_cats, x_nums = self.split(self.forward(x)[-1])
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(mask * self.emphasis + (1 - mask) * (1 - self.emphasis))

        cat_loss = weights[0] * torch.mul(w_cats, torch.nn.functional.binary_cross_entropy_with_logits(x_cats, y_cats, reduction='none'))
        num_loss = weights[1] * torch.mul(w_nums, torch.nn.functional.mse_loss(x_nums, y_nums, reduction='none'))
        reconstruction_loss = cat_loss.mean() + num_loss.mean()
        return reconstruction_loss

class DAE(BaseModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

    def _train(self,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               X_valid: Optional[pd.DataFrame] = None,
               y_valid: Optional[pd.Series] = None) -> DeepStackDAE:

        X, len_cat, len_num = get_dataset_with_cat_handler(X_train,
                                                           [*self.config.features.cat_features])
        dae_dl = get_dae_dataset(X,
                                 batch_size=self.config.model.data_batch_size,
                                 num_workers=self.config.model.num_workers)
        model = DeepStackDAE(len_cat,
                             len_num,
                             **self.config.model.params).cuda()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.model.optimizer.init_lr
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.config.model.optimizer.lr_gamma
        )
        earlystopper = EarlyStopping(mode='min',
                                     min_delta=self.config.model.earlystopper.min_delta,
                                     patience=self.config.model.earlystopper.patience,
                                     percentage=False,
                                     verbose=self.config.model.earlystopper.verbose)

        repeats = [len_cat, len_num]
        probas = [.8, self.config.model.noise_ratio]
        swap_probas = sum([[p] * r for p, r in zip(probas, repeats)], [])

        noise_maker = SwapNoiseMasker(swap_probas)

        for epoch in tqdm(range(self.config.model.iterations)):
            model.train()
            meter = AverageMeter()
            for i, x in enumerate(dae_dl):
                x = x.cuda()
                noisy_x, mask = noise_maker.apply(x)
                optimizer.zero_grad()
                loss = model.loss(noisy_x,
                                  x,
                                  mask,
                                  weights=[self.config.model.loss_weight.cat,
                                           self.config.model.loss_weight.num])
                loss.backward()
                optimizer.step()
                meter.update(loss.detach().cpu().numpy())
            scheduler.step()
            if epoch % 10 == 0:
                print(f'epoch: {epoch}, loss: {loss.detach().cpu().numpy()}')
            if earlystopper.step(meter.avg): break

        return model

    def train(self,
              train_x: pd.DataFrame,
              train_y: pd.Series) -> ModelResult:
        # model
        model = self._train(train_x, train_y)

        self.result = ModelResult(
            oof_preds=[],
            models=[model],
            scores={}
        )
        return self.result