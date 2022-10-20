import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

from .base_model import ModelResult

def load_model(config: DictConfig, model_name: str) -> ModelResult:
    """
    :param config: Hydra config
    :param model_name: model_name
    :return: ModelResult Object
    """

    model_path = Path(get_original_cwd()) / config.model.path / model_name

    with open(model_path, "rb") as output:
        model_result = pickle.load(output)

    return model_result

def inference(result: ModelResult,
              X_test: pd.DataFrame) -> np.ndarray:
    """
    :param result: ModelResult Object
    :param X_test: dataframe
    :return: predict probabilities for each class
    """

    folds = len(result.models)
    preds_proba = np.zeros((X_test.shape[0], ))

    for model in tqdm(result.models.values(), total=folds):
        preds_proba += (
            model.predict(X_test) / folds
            if isinstance(model, lgb.Booster)
            else model.predict(xgb.DMatrix(X_test)) / folds
            if isinstance(model, xgb.Booster)
            else model.predict_proba(X_test.to_numpy())[:, 1] / folds
        )

    assert len(preds_proba) == len(X_test)
    return preds_proba
