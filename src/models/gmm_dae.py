import math

import torch
import torch.nn.functional as F
from torch.nn.functional import softmax, log_softmax
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


def gaussian_pdf(x, mean, variance):
    # Compute the PDF of a Gaussian distribution
    eps = 1e-8
    variance = variance + eps
    return torch.exp(-0.5 * torch.square((x - mean)) / variance) / torch.sqrt(2.0 * torch.tensor(math.pi) * variance)


def gaussian_mixture_loss(y_pred, means, variances, weights, num_components):
    # Compute the PDFs for each component
    pdfs = [weights[i] * gaussian_pdf(y_pred, means[i], variances[i]) for i in range(num_components)]

    overall_pdf = torch.sum(torch.stack(pdfs), dim=0)
    eps = 1e-8
    overall_pdf = overall_pdf + eps
    # Compute the negative log-likelihood loss
    nll_loss = -torch.mean(torch.log(overall_pdf))

    return nll_loss


class GaussianDAE(torch.nn.Module):
    def __init__(self,
                 encoder_hidden_list,
                 decoder_hidden_list,
                 bottleneck_size,
                 num_components,
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
        self.num_components = num_components
        self.device = device
        self.post_encoding_input_size = len_cat + len_num

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
        self.gmm_estimation_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=z_dim, out_features=10),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio),
            torch.nn.Linear(in_features=10, out_features=self.num_components),
            torch.nn.Softmax(dim=1)
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

    def forward_gamma(self, x):
        enc = self.encoder_calibrate_layer(x)
        for encoder in self.encoder_list:
            enc = encoder(enc)
        z = self.bottleneck_layer(enc)
        dec = self.decoder_calibrate_layer(z)
        for decoder in self.decoder_list:
            dec = decoder(dec)
        dec = self.reconstruct_layer(dec)

        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x, dec)

        z = torch.cat([z, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

        gamma = self.gmm_estimation_layer(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        sum_gamma = torch.sum(gamma, dim=0)
        phi = sum_gamma / N
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)

        # z_minus_mu = N x K x D
        z_minus_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        # z_mu_outer = N x K x D x D
        z_mu_outer = z_minus_mu.unsqueeze(-1) * z_minus_mu.unsqueeze(-2)
        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov

    def compute_energy(self, z, gamma):
        phi, mu, cov = self.compute_gmm_params(z, gamma)

        # N x K x D
        z_minus_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        k, D, _ = cov.size()

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(self.num_components):
            # D x D
            cov_k = cov[i] + torch.eye(D).to(self.device) * eps
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            # cov_k_det_value = np.linalg.det(cov_k.data.cpu().numpy() * (2 * np.pi)) + eps
            _, cov_k_det_value = np.linalg.slogdet(cov_k.data.cpu().numpy() * (2 * np.pi))
            det_cov.append(cov_k_det_value)
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = (torch.from_numpy(np.float64(np.array(det_cov)))).to(self.device)
        print('det_cov', det_cov)
        # N x K
        exp_term_tmp = -0.5 * torch.sum(
            torch.sum(z_minus_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_minus_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)

        sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def loss(self, x, y, mask=None, weights=[3, 14]):
        if mask is None:
            mask = torch.ones(x.shape).to(x.device)

        enc, dec, z, gamma = self.forward_gamma(x)

        x_cats, x_nums = self.split(dec)
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(mask * self.emphasis + (1 - mask) * (1 - self.emphasis))

        cat_loss = weights[0] * torch.mul(w_cats, torch.nn.functional.binary_cross_entropy_with_logits(x_cats, y_cats,
                                                                                                       reduction='none'))
        num_loss = weights[1] * torch.mul(w_nums, torch.nn.functional.mse_loss(x_nums, y_nums, reduction='none'))

        if self.len_cat == 0:
            reconstruction_loss = num_loss.mean()
        else:
            reconstruction_loss = cat_loss.mean() + num_loss.mean()

        sample_energy, cov_diag = self.compute_energy(z, gamma)
        print(reconstruction_loss, sample_energy)
        total_loss = reconstruction_loss + sample_energy * 0.1# + cov_diag * 0.0005
        return total_loss, reconstruction_loss, sample_energy


class GMMDAE(BaseDAEModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

    def _get_model(self,
                   len_cat,
                   len_num,
                   device):
        model_name = self.config.model.name
        if model_name == 'gmm_dae':
            model = GaussianDAE(
                len_cat=len_cat,
                len_num=len_num,
                encoder_hidden_list=self.config.model.params.encoder_hidden_list,
                decoder_hidden_list=self.config.model.params.decoder_hidden_list,
                bottleneck_size=self.config.model.params.bottleneck_size,
                dropout_ratio=self.config.model.params.dropout_ratio,
                emphasis=self.config.model.params.emphasis,
                num_components=self.config.model.params.num_components,
                device=device
            ).to(device)
        return model

    def _train(self,
               train_dl: DataLoader,
               valid_dl: DataLoader,
               len_cat: int,
               len_num: int,
               device: torch.device) -> GaussianDAE:

        # model 가져오기
        model = self._get_model(len_cat,
                                len_num,
                                device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.model.optimizer.init_lr,
            # weight_decay=0.01
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
            meter_recon = AverageMeter()
            meter_eng = AverageMeter()
            # train
            for i, x in enumerate(train_dl):
                x = x.to(device)
                noisy_x, mask = noise_maker.apply(x)
                optimizer.zero_grad()
                loss, recon_loss, eng_loss = model.loss(noisy_x,
                                                        x,
                                                        mask,
                                                        weights=[self.config.model.loss_weight.cat,
                                                                 self.config.model.loss_weight.num])
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                meter.update(loss.detach().cpu().numpy())
                meter_recon.update(recon_loss.detach().cpu().numpy())
                meter_eng.update(eng_loss.detach().cpu().numpy())
            train_loss = meter.overall_avg
            train_recon_loss = meter_recon.overall_avg
            train_eng_loss = meter_eng.overall_avg
            metrics = {"train/train_loss": train_loss,
                       "train/train_recon_loss": train_recon_loss,
                       "train/train_eng_loss": train_eng_loss}
            # valid
            meter.reset()
            meter_recon.reset()
            meter_eng.reset()
            with torch.no_grad():
                for i, x in enumerate(valid_dl):
                    x = x.to(device)
                    noisy_x, mask = noise_maker.apply(x)
                    loss, recon_loss, eng_loss = model.loss(noisy_x,
                                      x,
                                      mask,
                                      weights=[self.config.model.loss_weight.cat,
                                               self.config.model.loss_weight.num])
                    meter.update(loss.detach().cpu().numpy())
                    meter_recon.update(recon_loss.detach().cpu().numpy())
                    meter_eng.update(eng_loss.detach().cpu().numpy())
            valid_loss = meter.overall_avg
            valid_recon_loss = meter_recon.overall_avg
            valid_eng_loss = meter_eng.overall_avg
            val_metrics = {"val/val_loss": valid_loss,
                           "val/val_recon_loss": valid_recon_loss,
                           "val/val_eng_loss": valid_eng_loss}

            wandb.log({**metrics, **val_metrics})

            if epoch % self.config.model.eval_verbose == 0:
                print('\repoch {:4d} - train loss {:6.4f} - train recon loss {:6.4f} - train eng loss {:6.4f} - valid loss {:6.4f}'.format(epoch, train_loss, train_recon_loss, train_eng_loss, valid_loss))

            scheduler.step(valid_loss)
            # checkpointing
            if valid_loss < best_score:
                best_score = valid_loss
                best_model = copy.deepcopy(model)
            # early stopping
            if earlystopper.step(valid_loss):
                break

        return best_model
