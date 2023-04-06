from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier

from .base_model import BaseModel


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
            "device_name": self.config.dataset.device
        }

    def _train(self,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               X_valid: Optional[pd.DataFrame] = None,
               y_valid: Optional[pd.Series] = None) -> TabNetClassifier:
        model = TabNetClassifier(**self._get_default_params())

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
                         CSSTabnetMetric]
        )

        return model