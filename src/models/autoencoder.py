import torch
import numpy as np
import pandas as pd
from typing import Callable, NoReturn, Optional, Tuple
from tqdm import tqdm
import copy
from torch.utils.data import Dataset, DataLoader
from ..utils.utils import SwapNoiseMasker, EarlyStopping, AverageMeter
from .base_model import BaseDAEModel
from .loss.RMSELoss import RMSELoss
import wandb


class DeepStackDAE(torch.nn.Module):
    def __init__(self,
                 len_cat,
                 len_num,
                 encoder_hidden_list,
                 decoder_hidden_list,
                 emphasis,
                 device):
        super().__init__()

        post_encoding_input_size = len_cat + len_num
        self.encoder_hidden_list = encoder_hidden_list
        self.decoder_hidden_list = decoder_hidden_list

        self.encoder_calibrate_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=post_encoding_input_size, out_features=self.encoder_hidden_list[0]),
            torch.nn.BatchNorm1d(self.encoder_hidden_list[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)

        self.encoder_list = []
        for prev, cur in zip(self.encoder_hidden_list[:-1], self.encoder_hidden_list[1:]):
            encoder = torch.nn.Sequential(
                torch.nn.Linear(in_features=prev, out_features=cur),
                torch.nn.BatchNorm1d(cur),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_ratio)
            ).to(device)
            self.encoder_list.append(encoder)

        self.decoder_calibrate_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.bottleneck_size, out_features=self.decoder_hidden_list[0]),
            torch.nn.BatchNorm1d(self.decoder_hidden_list[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)

        self.decoder_list = []
        for prev, cur in zip(self.decoder_hidden_list[:-1], self.decoder_hidden_list[1:]):
            decoder = torch.nn.Sequential(
                torch.nn.Linear(in_features=prev, out_features=cur),
                torch.nn.BatchNorm1d(cur),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_ratio)
            ).to(device)
            self.decoder_list.append(decoder)

        self.reconstruct_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.decoder_hidden_list[-1], out_features=post_encoding_input_size),
            torch.nn.BatchNorm1d(post_encoding_input_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)

        self.len_cat = len_cat
        self.len_num = len_num
        self.emphasis = emphasis

    def forward(self, x):
        x_list = []
        x = self.encoder_calibrate_layer(x)
        x_list.append(x)
        for encoder in self.encoder_list:
            x = encoder(x)
            x_list.append(x)

        x = self.decoder_calibrate_layer(x)
        x_list.append(x)
        for decoder in self.decoder_list:
            x = decoder(x)
            x_list.append(x)

        x = self.reconstruct_layer(x)
        x_list.append(x)
        return x_list

    def feature(self, x):
        return torch.cat(self.forward(x)[:-1], dim=1)

    def reconstructed_feature(self, x):
        return self.forward(x)[-1]

    def split(self, t):
        return torch.split(t, [self.len_cat, self.len_num], dim=1)

    def loss(self, x, y, mask=None, weights=[1, 1]):
        if mask is None:
            mask = torch.ones(x.shape).to(x.device)

        x_cats, x_nums = self.split(self.forward(x)[-1])
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(mask * self.emphasis + (1 - mask) * (1 - self.emphasis))

        cat_loss = weights[0] * torch.mul(w_cats, torch.nn.functional.binary_cross_entropy_with_logits(x_cats, y_cats,
                                                                                                       reduction='none'))
        num_loss = weights[1] * torch.mul(w_nums, torch.nn.functional.mse_loss(x_nums, y_nums, reduction='none'))
        if self.len_cat == 0:
            reconstruction_loss = num_loss.mean()
        else:
            reconstruction_loss = cat_loss.mean() + num_loss.mean()
        return reconstruction_loss


class DeepBottleneck(torch.nn.Module):
    def __init__(self,
                 encoder_hidden_list,
                 decoder_hidden_list,
                 bottleneck_size,
                 len_cat,
                 len_num,
                 dropout_ratio,
                 emphasis,
                 device):
        super().__init__()
        self.encoder_hidden_list = encoder_hidden_list
        self.decoder_hidden_list = decoder_hidden_list
        self.len_cat = len_cat
        self.len_num = len_num
        self.emphasis = emphasis
        self.bottleneck_size = bottleneck_size
        self.dropout_ratio = dropout_ratio
        self.device = device
        post_encoding_input_size = len_cat + len_num

        self.encoder_calibrate_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=post_encoding_input_size, out_features=self.encoder_hidden_list[0]),
            torch.nn.BatchNorm1d(self.encoder_hidden_list[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)

        self.encoder_list = []
        for prev, cur in zip(self.encoder_hidden_list[:-1], self.encoder_hidden_list[1:]):
            encoder = torch.nn.Sequential(
                torch.nn.Linear(in_features=prev, out_features=cur),
                torch.nn.BatchNorm1d(cur),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_ratio)
            ).to(device)
            self.encoder_list.append(encoder)

        self.bottleneck_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.encoder_hidden_list[-1], out_features=self.bottleneck_size),
            torch.nn.BatchNorm1d(self.bottleneck_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)

        self.decoder_calibrate_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.bottleneck_size, out_features=self.decoder_hidden_list[0]),
            torch.nn.BatchNorm1d(self.decoder_hidden_list[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)

        self.decoder_list = []
        for prev, cur in zip(self.decoder_hidden_list[:-1], self.decoder_hidden_list[1:]):
            decoder = torch.nn.Sequential(
                torch.nn.Linear(in_features=prev, out_features=cur),
                torch.nn.BatchNorm1d(cur),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_ratio)
            ).to(device)
            self.decoder_list.append(decoder)

        self.reconstruct_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.decoder_hidden_list[-1], out_features=post_encoding_input_size),
            torch.nn.BatchNorm1d(post_encoding_input_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)

    def forward_pass(self, x):
        x = self.encoder_calibrate_layer(x)
        for encoder in self.encoder_list:
            x = encoder(x)
        x = b = self.bottleneck_layer(x)
        x = self.decoder_calibrate_layer(x)
        for decoder in self.decoder_list:
            x = decoder(x)
        x = self.reconstruct_layer(x)
        return [b, x]

    def forward(self, x):
        return self.forward_pass(x)[1]

    def feature(self, x):
        return self.forward_pass(x)[0]

    def reconstructed_feature(self, x):
        return self.forward(x)

    def split(self, t):
        return torch.split(t, [self.len_cat, self.len_num], dim=1)

    def loss(self, x, y, mask=None, weights=[3, 14]):
        if mask is None:
            mask = torch.ones(x.shape).to(x.device)

        x_cats, x_nums = self.split(self.forward(x))
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(mask * self.emphasis + (1 - mask) * (1 - self.emphasis))

        # focal_loss = FocalLoss(size_average=False)
        cat_loss = weights[0] * torch.mul(w_cats, torch.nn.functional.binary_cross_entropy_with_logits(x_cats, y_cats, reduction='none'))
        num_loss = weights[1] * torch.mul(w_nums, torch.nn.functional.mse_loss(x_nums, y_nums, reduction='none'))
        # num_loss = weights[1] * torch.nn.functional.mse_loss(x_nums, y_nums, reduction='none')

        if self.len_cat == 0:
            reconstruction_loss = num_loss.mean()
        else:
            reconstruction_loss = cat_loss.mean() + num_loss.mean()
        return reconstruction_loss


class TransformerEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, feedforward_dim):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear_1 = torch.nn.Linear(embed_dim, feedforward_dim)
        self.linear_2 = torch.nn.Linear(feedforward_dim, embed_dim)
        self.layernorm_1 = torch.nn.LayerNorm(embed_dim)
        self.layernorm_2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x_in):
        attn_out, _ = self.attn(x_in, x_in, x_in)
        x = self.layernorm_1(x_in + attn_out)
        ff_out = self.linear_2(torch.nn.functional.relu(self.linear_1(x)))
        x = self.layernorm_2(x + ff_out)
        return x


class TransformerAutoEncoder(torch.nn.Module):
    def __init__(self,
                 len_cat,
                 len_num,
                 hidden_size=1024,
                 num_subspaces=8,
                 embed_dim=128,
                 num_heads=8,
                 dropout=0,
                 feedforward_dim=512,
                 emphasis=0.75,
                 mask_loss_weight=2):
        super().__init__()
        assert hidden_size == embed_dim * num_subspaces
        self.num_inputs = len_cat + len_num
        self.len_cat = len_cat
        self.len_num = len_num
        self.num_subspaces = num_subspaces
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.emphasis = emphasis
        self.mask_loss_weight = mask_loss_weight

        self.excite = torch.nn.Linear(in_features=self.num_inputs, out_features=hidden_size)
        self.encoder_1 = TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)
        self.encoder_2 = TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)
        self.encoder_3 = TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)

        self.mask_predictor = torch.nn.Linear(in_features=hidden_size, out_features=self.num_inputs)
        self.reconstructor = torch.nn.Linear(in_features=hidden_size + self.num_inputs, out_features=self.num_inputs)

    def divide(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.num_subspaces, self.embed_dim)).permute((1, 0, 2))
        return x

    def combine(self, x):
        batch_size = x.shape[1]
        x = x.permute((1, 0, 2)).reshape((batch_size, -1))
        return x

    def forward(self, x):
        x = torch.nn.functional.relu(self.excite(x))

        x = self.divide(x)
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x = self.combine(x3)

        predicted_mask = self.mask_predictor(x)
        reconstruction = self.reconstructor(torch.cat([x, predicted_mask], dim=1))
        return (x1, x2, x3), (reconstruction, predicted_mask)

    def split(self, t):
        return torch.split(t, [self.len_cat, self.len_num], dim=1)

    def feature(self, x):
        attn_outs, _ = self.forward(x)
        return torch.cat([self.combine(x) for x in attn_outs], dim=1)

    def reconstructed_feature(self, x):
        attn_outs, _ = self.forward(x)
        return torch.cat(attn_outs, dim=1)

    def loss(self, x, y, mask, weights=[3, 14], reduction='mean'):
        _, (reconstruction, predicted_mask) = self.forward(x)
        x_cats, x_nums = self.split(reconstruction)
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(mask * self.emphasis + (1 - mask) * (1 - self.emphasis))

        cat_loss = weights[0] * torch.mul(w_cats, torch.nn.functional.binary_cross_entropy_with_logits(x_cats, y_cats,
                                                                                                       reduction='none'))
        num_loss = weights[1] * torch.mul(w_nums, torch.nn.functional.mse_loss(x_nums, y_nums, reduction='none'))

        reconstruction_loss = torch.cat([cat_loss, num_loss], dim=1) if reduction == 'none' else cat_loss.mean() + num_loss.mean()
        mask_loss = self.mask_loss_weight * torch.nn.functional.binary_cross_entropy_with_logits(predicted_mask, mask,
                                                                                                 reduction=reduction)

        return reconstruction_loss + mask_loss if reduction == 'mean' else [reconstruction_loss, mask_loss]


class DAE(BaseDAEModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

    def _get_model(self,
                   len_cat,
                   len_num,
                   device):
        model_name = self.config.model.name
        if model_name == 'deepstack_dae':
            model = DeepStackDAE(
                len_cat=len_cat,
                len_num=len_num,
                encoder_hidden_list=self.config.model.params.encoder_hidden_list,
                decoder_hidden_list=self.config.model.params.decoder_hidden_list,
                emphasis=self.config.model.params.emphasis,
                device=device
            ).to(device)
        elif model_name == 'bottleneck_dae':
            model = DeepBottleneck(
                len_cat=len_cat,
                len_num=len_num,
                encoder_hidden_list=self.config.model.params.encoder_hidden_list,
                decoder_hidden_list=self.config.model.params.decoder_hidden_list,
                bottleneck_size=self.config.model.params.bottleneck_size,
                dropout_ratio=self.config.model.params.dropout_ratio,
                emphasis=self.config.model.params.emphasis,
                device=device
            ).to(device)
        elif model_name == 'transformer_dae':
            model = TransformerAutoEncoder(
                len_cat=len_cat,
                len_num=len_num,
                hidden_size=self.config.model.hidden_size,
                num_subspaces=self.config.model.num_subspaces,
                embed_dim=self.config.model.embed_dim,
                num_heads=self.config.model.num_heads,
                dropout=self.config.model.dropout,
                feedforward_dim=self.config.model.feedforward_dim,
                emphasis=self.config.model.emphasis,
                mask_loss_weight=self.config.model.mask_loss_weight
            ).to(device)
        return model

    def _train(self,
               train_dl: DataLoader,
               valid_dl: DataLoader,
               len_cat: int,
               len_num: int,
               device: torch.device) -> DeepStackDAE:

        # model 가져오기
        model = self._get_model(len_cat,
                                len_num,
                                device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.model.optimizer.init_lr
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.config.model.scheduler.gamma
        )
        earlystopper = EarlyStopping(mode='min',
                                     min_delta=self.config.model.earlystopper.min_delta,
                                     patience=self.config.model.earlystopper.patience,
                                     percentage=False,
                                     verbose=self.config.model.earlystopper.verbose)

        repeats = [len_cat, len_num]
        probas = [self.config.model.noise_ratio.cat,
                  self.config.model.noise_ratio.num]
        swap_probas = sum([[p] * r for p, r in zip(probas, repeats)], [])
        noise_maker = SwapNoiseMasker(swap_probas)
        best_score = float('inf')
        for epoch in tqdm(range(self.config.model.iterations)):
            model.train()
            meter = AverageMeter()
            # train
            for i, x in enumerate(train_dl):
                x = x.to(device)
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
            train_loss = meter.overall_avg
            metrics = {"train/train_loss": train_loss}
            # valid
            meter.reset()
            model.eval()
            with torch.no_grad():
                for i, x in enumerate(valid_dl):
                    x = x.to(device)
                    noisy_x, mask = noise_maker.apply(x)
                    loss = model.loss(noisy_x,
                                      x,
                                      mask,
                                      weights=[self.config.model.loss_weight.cat,
                                               self.config.model.loss_weight.num])
                    meter.update(loss.detach().cpu().numpy())
            valid_loss = meter.overall_avg
            val_metrics = {"val/val_loss": valid_loss}
            wandb.log({**metrics, **val_metrics})

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
