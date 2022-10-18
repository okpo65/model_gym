import gc
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold

from .base_boosting import BaseModel, ModelResult

class TabNetTrainer(BaseModel):
    def __init__(self,
                 params: Optional[Dict[str, Any]],
                 cat_idxs: Optional[List[int]]=None,
                 cat_dims: Optional[List[int]]=None,
                 search: bool=False,
                 **kwargs,):
        self.params = params
        self.search = search
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        super().__init__(**kwargs)

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=self.config.model.params.optimizer_lr),
            "scheduler_params": {"step_size": self.config.model.params.scheduler_step_size,
                                 "gamma": self.config.model.params.scheduler_gamma},
            "scheduler_fn": torch.optim.lr_scheduler.StepLR,
            "mask_type": self.config.model.params.mask_type,  # "sparsemax"
        }

    def _train(self,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               X_valid: Optional[pd.DataFrame] = None,
               y_valid: Optional[pd.Series] = None) -> TabNetClassifier:
        model = TabNetClassifier(**self._get_default_params())

        model.fit(
            **self.config.model.fit_params,
            X_train=X_train.values,
            y_train=y_train.values,
            eval_set=[
                (X_train.to_numpy(), y_train.to_numpy()),
                (X_valid.to_numpy(), y_valid.to_numpy()),
            ],
            eval_name=["train", "val"],
            eval_metric=["logloss"]
        )

        return model