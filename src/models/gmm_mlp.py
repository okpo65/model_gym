import copy
from typing import NoReturn

import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from src.models.base_model import BaseDLModel
from src.utils.utils import EarlyStopping, AverageMeter


def gaussian_mixture_binary_loss(y_preds, y_true, means, variances, weights):
    # compute the binary cross-entropy loss
    mse_loss = torch.nn.functional.mse_loss(y_preds, y_true)

    # compute the Gaussian mixture loss
    num_components = means.shape[1]
    num_samples = y_preds.shape[0]

    # repeat the logits and y_true tensors for each component
    y_preds = y_preds.repeat(1, num_components).view(num_samples*num_components, -1)
    y_true = y_true.repeat(1, num_components).view(num_samples*num_components, -1)

    # repeat the weights tensor for each sample
    weights = weights.repeat(num_samples, 1)

    # compute the log-likelihood of each component
    normal_dist = torch.distributions.normal.Normal(loc=means, scale=torch.sqrt(variances))
    log_likelihood = normal_dist.log_prob(y_preds.unsqueeze(-1)).sum(dim=-1)

    # compute the log-likelihood of the mixture
    log_prob = torch.logsumexp(torch.log(weights) + log_likelihood, dim=1)
    gm_loss = -log_prob.mean()

    # compute the total loss
    loss = mse_loss + gm_loss
    return loss



class GaussianMLP(torch.nn.Module):

    def __init__(self,
                 len_cat,
                 len_num,
                 hidden_size_list,
                 dropout_ratio,
                 num_components,
                 device=torch.device('cpu')):
        super().__init__()

        self.len_total = len_cat + len_num
        self.device = device
        self.hidden_size_list = hidden_size_list
        self.dropout_ratio = dropout_ratio
        self.num_components = num_components

        self.calibrate_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.len_total, out_features=self.hidden_size_list[0]),
            torch.nn.BatchNorm1d(self.hidden_size_list[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)

        self.hidden_layer_list = []
        for prev, cur in zip(self.hidden_size_list[:-1], self.hidden_size_list[1:]):
            mlp_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features=prev, out_features=cur),
                torch.nn.BatchNorm1d(cur),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_ratio)
            ).to(device)
            self.hidden_layer_list.append(mlp_layer)

        self.last_linear = torch.nn.Linear(self.hidden_size_list[-1], 1).to(device)

        self.means = torch.nn.Parameter(torch.randn(1, self.num_components))
        self.variances = torch.nn.Parameter(torch.randn(1, self.num_components))
        self.weights = torch.nn.Parameter(torch.ones(1, self.num_components) / self.num_components)

    def forward(self, x):
        x = self.calibrate_layer(x)
        for hidden_layer in self.hidden_layer_list:
            x = hidden_layer(x)
        x = self.last_linear(x)
        return x, self.means, self.variances, self.weights

    def loss(self, x, y, means, variances, weights):
        mlp_loss = gaussian_mixture_binary_loss(x, y, means, variances, weights)
        return mlp_loss

    def predict(self, test_dl: DataLoader) -> np.ndarray:
        self.eval()
        predictions = []
        with torch.no_grad():
            for i, x in enumerate(test_dl):
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                prediction = self.forward(x)
                predictions.append(prediction.detach().cpu().numpy())
        predictions = np.concatenate(predictions).reshape(-1)
        return predictions

class GMMMLP(BaseDLModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

    def _train(self,
               train_dl: DataLoader,
               valid_dl: DataLoader,
               len_cat: int,
               len_num: int,
               device: torch.device) -> GaussianMLP:

        model = GaussianMLP(
            len_cat=len_cat,
            len_num=len_num,
            dropout_ratio=self.config.model.params.dropout_ratio,
            hidden_size_list=self.config.model.params.hidden_size_list,
            num_components=self.config.model.params.num_components,
            device=device,
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

        for epoch in tqdm(range(self.config.model.iterations)):
            model.train()
            meter = AverageMeter()
            # train
            for i, (x, target) in enumerate(train_dl):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                prediction, means, variances, weights = model.forward(x)
                loss = model.loss(prediction, target.unsqueeze(1), means, variances, weights)
                loss.backward()
                optimizer.step()
                meter.update(loss.detach().cpu().numpy())
            train_loss = meter.overall_avg
            metrics = {"train/train_loss": train_loss,
                       "train/learning_rate": optimizer.param_groups[0]['lr']}
            # valid
            meter.reset()
            model.eval()
            predictions = []
            with torch.no_grad():
                for i, (x, target) in enumerate(valid_dl):
                    x, target = x.to(device), target.to(device)
                    prediction, means, variances, weights = model.forward(x)
                    loss = model.loss(prediction, target.unsqueeze(1), means, variances, weights)

                    prediction = prediction[:, -1]
                    predictions.append(prediction.detach().cpu().numpy())
                    meter.update(loss.detach().cpu().numpy())
            predictions = np.concatenate(predictions)
            valid_loss = meter.overall_avg
            valid_metric_value = self.metric(predictions, valid_dl.dataset.y)
            val_metrics = {"val/val_loss": valid_loss,
                           "val/val_metric": valid_metric_value}
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