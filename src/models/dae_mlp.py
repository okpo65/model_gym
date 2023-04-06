import copy
from typing import NoReturn

import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from src.models.base_model import BaseDLModel
from src.models.utils.utils import SwapNoiseMasker
from src.utils.utils import EarlyStopping, AverageMeter


class DAEClassifier(torch.nn.Module):
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
        self.post_encoding_input_size = len_cat + len_num
        self.upper_bound = 10.5
        self.lower_bound = 0

        self.encoder_calibrate_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.post_encoding_input_size, out_features=self.encoder_hidden_list[0]),
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
            torch.nn.Linear(in_features=self.decoder_hidden_list[-1], out_features=self.post_encoding_input_size),
            torch.nn.BatchNorm1d(self.post_encoding_input_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)

        z_dim = self.bottleneck_size + 2
        self.mlp_estimation_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=z_dim, out_features=100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio),
            torch.nn.Linear(in_features=100, out_features=50),
            torch.nn.BatchNorm1d(50),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=50, out_features=1),
        ).to(device)

    def forward_pass(self, x):
        x = self.encoder_calibrate_layer(x)
        for encoder in self.encoder_list:
            x = encoder(x)
        x = z = self.bottleneck_layer(x)
        x = self.decoder_calibrate_layer(x)
        for decoder in self.decoder_list:
            x = decoder(x)
        x = self.reconstruct_layer(x)

        return [z, x]

    def forward(self, x):
        return self.forward_pass(x)[1]

    def feature(self, x):
        return self.forward_pass(x)[0]

    def reconstructed_feature(self, x):
        return self.forward(x)

    def split(self, t):
        return torch.split(t, [self.len_cat, self.len_num], dim=1)

    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward_dae(self, x):
        enc = self.encoder_calibrate_layer(x)
        for encoder in self.encoder_list:
            enc = encoder(enc)
        z = self.bottleneck_layer(enc)
        dec = self.decoder_calibrate_layer(z)
        for decoder in self.decoder_list:
            dec = decoder(dec)
        dec = self.reconstruct_layer(dec)

        return z, enc, dec

    def forward_mlp(self, z):
        pred = self.mlp_estimation_layer(z)
        pred = torch.sigmoid(pred * (self.upper_bound - self.lower_bound) + self.lower_bound)
        return pred

    def recon_loss(self, x, y, mask=None):
        if mask is None:
            mask = torch.ones(x.shape).to(x.device)

        x_cats, x_nums = self.split(x)
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(mask * self.emphasis + (1 - mask) * (1 - self.emphasis))

        num_loss = torch.mul(w_nums, torch.nn.functional.mse_loss(x_nums, y_nums, reduction='none'))
        reconstruction_loss = num_loss
        return reconstruction_loss

    def mlp_loss(self, pred, y_true):
        mlp_loss = torch.nn.functional.binary_cross_entropy(pred, y_true)
        return mlp_loss

    def predict(self, test_dl: DataLoader) -> np.ndarray:
        self.eval()
        predictions = []
        with torch.no_grad():
            for i, x in enumerate(test_dl):
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                z, _, dec = self.forward_dae(x)
                recon_loss = self.recon_loss(dec, x)
                rec_cosine = torch.nn.functional.cosine_similarity(x, dec, dim=1)
                z = torch.cat([z, recon_loss.mean(axis=1).unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
                prediction = self.forward_mlp(z)
                predictions.append(prediction.detach().cpu().numpy())
        predictions = np.concatenate(predictions).reshape(-1)
        return predictions

class DAEMLP(BaseDLModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

    def _train(self,
               train_dl: DataLoader,
               valid_dl: DataLoader,
               len_cat: int,
               len_num: int,
               device: torch.device) -> DAEClassifier:

        model = DAEClassifier(
            len_cat=len_cat,
            len_num=len_num,
            encoder_hidden_list=self.config.model.params.encoder_hidden_list,
            decoder_hidden_list=self.config.model.params.decoder_hidden_list,
            bottleneck_size=self.config.model.params.bottleneck_size,
            dropout_ratio=self.config.model.params.dropout_ratio,
            emphasis=self.config.model.params.emphasis,
            device=device
        ).to(device)

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

        repeats = [len_cat, len_num]
        probas = [self.config.model.noise_ratio.cat,
                  self.config.model.noise_ratio.num]
        swap_probas = sum([[p] * r for p, r in zip(probas, repeats)], [])
        noise_maker = SwapNoiseMasker(swap_probas)

        for epoch in tqdm(range(self.config.model.iterations)):
            model.train()
            meter = AverageMeter()
            meter_recon = AverageMeter()
            meter_mlp = AverageMeter()
            # train
            for i, (x, target) in enumerate(train_dl):
                x, target = x.to(device), target.to(device)
                noisy_x, mask = noise_maker.apply(x)
                optimizer.zero_grad()
                z, _, dec = model.forward_dae(noisy_x)
                recon_loss = model.recon_loss(dec, x, mask)
                rec_cosine = torch.nn.functional.cosine_similarity(x, dec, dim=1)
                z = torch.cat([z, recon_loss.mean(axis=1).unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
                recon_loss = recon_loss.mean()
                pred = model.forward_mlp(z)
                mlp_loss = model.mlp_loss(pred=pred, y_true=target.unsqueeze(1))
                total_loss = recon_loss * 0.5 + mlp_loss * 0.5
                total_loss.backward()
                optimizer.step()
                meter.update(total_loss.detach().cpu().numpy())
                meter_recon.update(recon_loss.detach().cpu().numpy())
                meter_mlp.update(mlp_loss.detach().cpu().numpy())
            train_loss = meter.overall_avg
            train_recon_loss = meter_recon.overall_avg
            train_mlp_loss = meter_mlp.overall_avg

            metrics = {"train/train_loss": train_loss,
                       "train/recon_loss": train_recon_loss,
                       "train/mlp_loss": train_mlp_loss}
            # valid
            meter.reset()
            meter_recon.reset()
            meter_mlp.reset()
            model.eval()
            predictions = []
            with torch.no_grad():
                for i, (x, target) in enumerate(valid_dl):
                    x, target = x.to(device), target.to(device)
                    noisy_x, mask = noise_maker.apply(x)
                    z, _, dec = model.forward_dae(noisy_x)
                    recon_loss = model.recon_loss(dec, x, mask)
                    rec_cosine = torch.nn.functional.cosine_similarity(x, dec, dim=1)
                    z = torch.cat([z, recon_loss.mean(axis=1).unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
                    recon_loss = recon_loss.mean()
                    pred = model.forward_mlp(z)
                    mlp_loss = model.mlp_loss(pred=pred, y_true=target.unsqueeze(1))
                    total_loss = recon_loss * 0.3 + mlp_loss * 0.7

                    predictions.append(pred.detach().cpu().numpy())
                    meter.update(total_loss.detach().cpu().numpy())
                    meter_recon.update(recon_loss.detach().cpu().numpy())
                    meter_mlp.update(mlp_loss.detach().cpu().numpy())

            predictions = np.concatenate(predictions).squeeze()
            valid_loss = meter.overall_avg
            valid_metric_value = self.metric(predictions, valid_dl.dataset.y)
            valid_recon_loss = meter_recon.overall_avg
            valid_mlp_loss = meter_mlp.overall_avg
            val_metrics = {"val/val_loss": valid_loss,
                           "val/val_metric": valid_metric_value,
                           "val/recon_loss": valid_recon_loss,
                           "val/mlp_loss": valid_mlp_loss}
            wandb.log({**metrics, **val_metrics})

            if epoch % self.config.model.eval_verbose == 0:
                print('\repoch {:4d} - train loss {:6.4f} - valid loss {:6.4f} - valid metric {:4.4f}'.format(epoch, train_loss, valid_loss, valid_metric_value))

            scheduler.step(valid_loss)
            # checkpointing
            if valid_loss < best_score:
                best_score = valid_loss
                best_model = copy.deepcopy(model)
            # early stopping
            if earlystopper.step(valid_loss):
                break


        return best_model