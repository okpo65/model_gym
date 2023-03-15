import torch
import numpy as np
import pandas as pd
from typing import NoReturn
from tqdm import tqdm
import copy
from torch.utils.data import DataLoader
from .utils.utils import EarlyStopping, AverageMeter
from .base_model import BaseDLModel
import wandb
# from src.models.loss.FocalLoss import FocalLoss


class DeepStackMLP(torch.nn.Module):
    def __init__(self,
                 len_cat,
                 len_num,
                 dropout_ratio,
                 hidden_size,
                 lower_bound=0,
                 upper_bound=10.5,
                 device=torch.device('cpu')):
        super().__init__()

        self.len_total = len_cat + len_num
        self.hidden_size = hidden_size
        self.device = device
        self.half_hidden_size = int(self.hidden_size / 2)
        self.dropout_ratio = dropout_ratio
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.len_total, out_features=self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hidden_size, out_features=self.half_hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.half_hidden_size),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)
        self.layer_3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.half_hidden_size, out_features=self.half_hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.half_hidden_size),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)
        self.layer_4 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.half_hidden_size, out_features=int(self.half_hidden_size/2)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(int(self.half_hidden_size/2)),
            torch.nn.Dropout(self.dropout_ratio)
        ).to(device)
        self.last_linear = torch.nn.Linear(self.half_hidden_size, 1).to(device)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def net(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        # x = self.layer_4(x)
        return self.last_linear(x)

    def forward(self, x):
        return torch.sigmoid(self.net(x) * (self.upper_bound - self.lower_bound) + self.lower_bound)

    def loss(self, x, y):
        mlp_loss = torch.nn.functional.binary_cross_entropy(x, y)
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

    # MC Dropout
    def predict_uncertainty(self, test_dl: DataLoader, n_process):
        """
        :param test_dl: Test dataloader
        :param n_process: number of times how many mc dropout process
        :return:
        """
        self.eval()
        self.train()
        total_predictions = []
        with torch.no_grad():
            for _ in tqdm(range(n_process)):
                predictions = []
                for i, x in enumerate(test_dl):
                    if isinstance(x, list):
                        x = x[0]
                    x = x.to(self.device)
                    prediction = self.forward(x)
                    predictions.append(prediction.detach().cpu().numpy())
                predictions = np.concatenate(predictions).reshape(-1)
                if total_predictions == []:
                    total_predictions = predictions
                else:
                    total_predictions = np.vstack([total_predictions, predictions])
            mean = np.mean(total_predictions, axis=0)
            std = np.std(total_predictions, axis=0)

        return mean, std

        # MC Dropout
    def predict_uncertainty_2(self, X_test: pd.DataFrame, n_process):
        """
        :param test_dl: Test dataloader
        :param n_process: number of times how many mc dropout process
        :return:
        """
        self.eval()
        self.train()
        total_predictions = []
        with torch.no_grad():
            for _ in tqdm(range(n_process)):
                predictions = []
                input_value = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(self.device)
                input_value = input_value.to(self.device)
                prediction = self.forward(input_value)
                predictions.append(prediction.detach().cpu().numpy())

                predictions = np.concatenate(predictions).reshape(-1)

                if total_predictions == []:
                    total_predictions = predictions
                else:
                    total_predictions = np.vstack([total_predictions, predictions])
            mean = np.mean(total_predictions, axis=0)
            std = np.std(total_predictions, axis=0)

        return mean, std

class DeepStackMLP_V2(torch.nn.Module):
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
        self.half_half_hidden_size = int(self.half_hidden_size / 2)
        self.dropout_ratio = dropout_ratio
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.len_total, out_features=self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.Dropout(self.dropout_ratio)
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.Dropout(self.dropout_ratio)
        )
        self.layer_3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.Dropout(self.dropout_ratio)
        )
        self.layer_4 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.Dropout(self.dropout_ratio)
        )
        self.last_linear = torch.nn.Linear(self.hidden_size, 1)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def net(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        # x = self.layer_4(x)
        return self.last_linear(x)

    def forward(self, x):
        return torch.sigmoid(self.net(x) * (self.upper_bound - self.lower_bound) + self.lower_bound)

    def loss(self, x, y, weights=[3, 14]):
        mlp_loss = torch.nn.functional.binary_cross_entropy(x, y)
        return mlp_loss

    def split(self, t):
        return torch.split(t, [self.len_cat, self.len_num], dim=1)

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

    # MC Dropout
    def predict_uncertainty(self, test_dl: DataLoader, n_process):
        """
        :param test_dl: Test dataloader
        :param n_process: number of times how many mc dropout process
        :return:
        """
        self.eval()
        self.train()
        total_predictions = []
        with torch.no_grad():
            for _ in tqdm(range(n_process)):
                predictions = []
                for i, x in enumerate(test_dl):
                    if isinstance(x, list):
                        x = x[0]
                    x = x.to(device)
                    prediction = self.forward(x)
                    predictions.append(prediction.detach().cpu().numpy())
                predictions = np.concatenate(predictions).reshape(-1)
                if total_predictions == []:
                    total_predictions = predictions
                else:
                    total_predictions = np.vstack([total_predictions, predictions])
            mean = np.mean(total_predictions, axis=0)
            std = np.std(total_predictions, axis=0)

        return mean, std

class MLP(BaseDLModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

    def _train(self,
               train_dl: DataLoader,
               valid_dl: DataLoader,
               len_cat: int,
               len_num: int,
               device: torch.device) -> DeepStackMLP:

        model = DeepStackMLP(
            len_cat=len_cat,
            len_num=len_num,
            dropout_ratio=self.config.model.params.dropout_ratio,
            hidden_size=self.config.model.params.hidden_size,
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

        for epoch in tqdm(range(self.config.model.iterations)):
            model.train()
            meter = AverageMeter()
            # train
            for i, (x, target) in enumerate(train_dl):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                loss = model.loss(model.forward(x), target.unsqueeze(1))
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
                    prediction = model.forward(x)
                    loss = model.loss(prediction, target.unsqueeze(1))
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