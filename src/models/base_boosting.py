import gc
import logging
import pickle
import warnings
from abc import ABCMeta, abstractclassmethod, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, NoReturn, Optional, List
from torch.utils.data import Dataset, DataLoader

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from ..dataset.dataset import DataContainer
from wandb.sklearn import plot_feature_importances

@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: Dict[str, Any]
    scores: Dict[str, Dict[str, float]]


class BaseDAEModel(metaclass=ABCMeta):
    def __init__(self,
                 config: DictConfig):
        self.config = config
        self.result = None

    @abstractmethod
    def _train(self,
               train_dl: DataLoader,
               valid_dl: DataLoader,
               len_cat: int,
               len_num: int) -> NoReturn:
        raise NotImplementedError

    def save_model(self) -> NoReturn:
        """
        Save Model
        """

        model_path = (
            Path(get_original_cwd()) / self.config.model.path / self.config.model.result
        )

        with open(model_path, 'wb') as output:
            pickle.dump(self.result, output, pickle.HIGHEST_PROTOCOL)

    def train(self,
              train_cont: DataContainer) -> ModelResult:
        train_dl, valid_dl = train_cont.get_dae_dataset(batch_size=self.config.model.batch_size,
                                                        num_workers=self.config.model.num_workers)
        len_cat, len_num = train_cont.len_cat, train_cont.len_num

        model = self._train(train_dl, valid_dl, len_cat, len_num)

        self.result = ModelResult(
            oof_preds=np.array(),
            models=[model],
            scores={}
        )
        return self.result

class BaseDLModel(metaclass=ABCMeta):
    def __init__(self,
                 config: DictConfig,
                 metric: Callable[[np.ndarray, np.ndarray], float]):
        self.config = config
        self.metric = metric
        self.result = None

    @abstractmethod
    def _train(self,
               train_dl: DataLoader,
               valid_dl: DataLoader,
               len_cat: int,
               len_num: int) -> NoReturn:
        raise NotImplementedError


    def save_model(self) -> NoReturn:
            """
            Save Model
            """

            model_path = (
                Path(get_original_cwd()) / self.config.model.path / self.config.model.result
            )

            with open(model_path, 'wb') as output:
                pickle.dump(self.result, output, pickle.HIGHEST_PROTOCOL)

    def train(self,
              train_cont: DataContainer):
        train_dl, valid_dl = train_cont.get_dae_dataset(batch_size=self.config.model.batch_size,
                                                        num_workers=self.config.model.num_workers)
        len_cat, len_num = train_cont.len_cat, train_cont.len_num
        model = self._train(train_dl, valid_dl, len_cat, len_num)

        self.result = ModelResult(
            oof_preds=np.array(),
            models=[model],
            scores={}
        )
        return self.result

class BaseModel(metaclass=ABCMeta):
    def __init__(self,
                 config: DictConfig,
                 metric: Callable[[np.ndarray, np.ndarray], float],
                 search: bool = False):
        self.config = config
        self.metric = metric
        self.search = search
        self._max_score = 0.0
        self._num_fold_iter = 0
        self.result = None

    @abstractmethod
    def _train(self,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               X_valid: Optional[pd.DataFrame] = None,
               y_valid: Optional[pd.Series] = None) -> NoReturn:
        raise NotImplementedError

    def save_model(self) -> NoReturn:
        """
        Save Model
        """

        model_path = (
            Path(get_original_cwd()) / self.config.model.path / self.config.model.result
        )

        with open(model_path, 'wb') as output:
            pickle.dump(self.result, output, pickle.HIGHEST_PROTOCOL)

    def train(self,
              train_cont: DataContainer) -> ModelResult:
        """
        :param train_x: train dataset
        :param train_y: target dataset
        :return: Model result
        """
        models = dict()
        scores = dict()
        folds = self.config.model.fold
        seed = self.config.dataset.seed

        train_x, train_y = train_cont.get_dataframe()

        str_kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = str_kf.split(train_x, train_y)
        oof_preds = np.zeros(len(train_x))

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            self._max_score = 0.0
            self._num_fold_iter = fold

            X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

            # need wandb logger

            # model
            model = self._train(X_train, y_train, X_valid, y_valid)
            models[f"fold_{fold}"] = model

            # validation
            oof_preds[valid_idx] = (
                model.predict(X_valid) if isinstance(model, lgb.Booster) else model.predict(xgb.DMatrix(X_valid)) if isinstance(model, xgb.Booster) else model.predict_proba(X_valid.to_numpy())[:, 1]
            )
            # score
            score = self.metric(y_valid, oof_preds[valid_idx])
            print(f"fold_metric: {score}")
            scores[f"fold_{fold}"] = score

            del X_train, X_valid, y_train, y_valid, model
            gc.collect()

        oof_score = self.metric(train_y, oof_preds)
        self.result = ModelResult(
            oof_preds=oof_preds,
            models=models,
            scores={"oof_score":oof_score, "KFold_scores": scores}
        )
        return self.result