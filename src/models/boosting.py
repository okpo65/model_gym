from typing import Callable, NoReturn, Optional, Tuple

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

from .base_model import BaseModel
from wandb.lightgbm import wandb_callback, log_summary

class LGBMTrainer(BaseModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

    def _train(self,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               X_valid: Optional[pd.DataFrame]=None,
               y_valid: Optional[pd.DataFrame]=None) -> lgb.Booster:

        train_set = lgb.Dataset(
            data=X_train,
            label=y_train,
            categorical_feature=[],
        )
        valid_set = lgb.Dataset(
            data=X_valid,
            label=y_valid,
            categorical_feature=[],
        )

        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(self.config.model.params),
            verbose_eval=self.config.model.verbose,
            num_boost_round=self.config.model.num_boost_round,
            callbacks=[wandb_callback()]
            # feval=lgb_amex_metric,
            # fobj=self._weighted_logloss if self.config.model.loss.is_customized else None,
        )

        return model

class CatBoostTrainer(BaseModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

    def _train(self,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               X_valid: Optional[pd.DataFrame] = None,
               y_valid: Optional[pd.Series] = None) -> CatBoostClassifier:

        train_data = Pool(
            data=X_train,
            label=y_train,
            cat_features=[]
        )
        valid_data = Pool(
            data=X_valid,
            label=y_valid,
            cat_features=[]
        )
        model = CatBoostClassifier(
            **self.config.model.params
        )
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            verbose=self.config.model.verbose
        )
        return model

class XGBoostTrainer(BaseModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

    def _train(self,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               X_valid: Optional[pd.DataFrame] = None,
               y_valid: Optional[pd.Series] = None) -> xgb.Booster:
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
        watchlist = [(dvalid, "eval")]

        model = xgb.train(
            dict(self.config.model.params),
            dtrain=dtrain,
            num_boost_round=self.config.model.num_boost_round,
            evals=watchlist,
            verbose_eval=self.config.model.verbose,
        )

        return model
