from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from .infer import load_pretrained_model
from ..models.tabnet_model.metrics import Metric
from ..models.tabnet_model.tab_model import TabNetClassifier
from ..models.tabnet_model.pretraining import TabNetPretrainer

from .base_model import BaseModel, BaseTabNetPretrainModel

# using amex metric to evaluate tabnet
from ..evaluation.evaluation import CSSMetric


class CSSTabnetMetric(Metric):

    def __init__(self):
        self._name = 'css_tabnet'
        self._maximize = True

    def __call__(self, y_true, y_pred):
        ks = CSSMetric().get_ks(y_pred[:, 1], y_true)
        return max(ks, 0.)

class TabNetTrainer(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=self.config.model.params.optimizer_lr),
            "scheduler_params": {"step_size": self.config.model.params.scheduler_step_size,
                                 "gamma": self.config.model.params.scheduler_gamma},
            "scheduler_fn": torch.optim.lr_scheduler.StepLR,
            "mask_type": self.config.model.params.mask_type,  # "sparsemax"
            "device_name": self.config.dataset.device,
            "n_d": self.config.model.params.n_d,
            "n_a": self.config.model.params.n_a,
            "n_steps": self.config.model.params.n_steps,
            "gamma": self.config.model.params.gamma,
            "lambda_sparse": self.config.model.params.lambda_sparse,
        }

    def _train(self,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               X_valid: Optional[pd.DataFrame] = None,
               y_valid: Optional[pd.Series] = None) -> TabNetClassifier:
        model = TabNetClassifier(**self._get_default_params())

        pretrained_model = None
        if "pretrained_model" in self.config.model.keys():
            pretrained_model = load_pretrained_model(self.config.model.pretrained_model)

        model.fit(
            **self.config.model.fit_params,
            X_train=X_train.to_numpy(),
            y_train=y_train,
            eval_set=[
                (X_train.to_numpy(), y_train),
                (X_valid.to_numpy(), y_valid),
            ],
            eval_name=["train", "val"],
            eval_metric=[self.config.model.eval_metric,
                         CSSTabnetMetric],
            from_unsupervised=pretrained_model
        )

        return model


class TabNetPretrainModel(BaseTabNetPretrainModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=self.config.model.params.optimizer_lr),
            "scheduler_params": {"step_size": self.config.model.params.scheduler_step_size,
                                 "gamma": self.config.model.params.scheduler_gamma},
            "scheduler_fn": torch.optim.lr_scheduler.StepLR,
            "mask_type": self.config.model.params.mask_type,  # "sparsemax"
            "device_name": self.config.dataset.device,
            "n_d": self.config.model.params.n_d,
            "n_a": self.config.model.params.n_a,
            "n_steps": self.config.model.params.n_steps,
            "gamma": self.config.model.params.gamma,
            "lambda_sparse": self.config.model.params.lambda_sparse,
        }

    def _train(self,
               X_train: pd.DataFrame,
               X_valid: Optional[pd.DataFrame],
               device: torch.device) -> TabNetPretrainer:
        model = TabNetPretrainer(**self._get_default_params())

        model.fit(
            **self.config.model.fit_params,
            X_train=X_train.to_numpy(),
            eval_set=[
                X_valid.to_numpy()
            ],
            pretraining_ratio=0.5
        )

        return model