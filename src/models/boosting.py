from typing import Callable, NoReturn, Optional, Tuple

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

from .base_model import BaseModel

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
            random_state=self.config.model.seed,
            cat_features=[],
            task_type='GPU',
            devices='1',
            **self.config.model.params
        )
        model.fit(
            train_data,
            eval_set=valid_data,
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
        watchlist = [(dtrain, "train"), (dvalid, "valid")]

        model = xgb.train(
            dict(self.config.model.params),
            dtrain=dtrain,
            evals=watchlist,
            num_boost_round=self.config.model.num_boost_round,
            early_stopping_rounds=self.config.model.early_stopping_rounds,
            verbose_eval=self.config.model.verbose,
        )

        return model
