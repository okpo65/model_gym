import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from ..utils.utils import SwapNoiseMasker
import torch
from tqdm import tqdm
from ..dataset.dataset import DataContainer
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
            if isinstance(model, CatBoostClassifier)
            else model.predict(X_test.to_numpy())
        )
    assert len(preds_proba) == len(X_test)

    return preds_proba

def inference_mlp(result: ModelResult,
                  test_dl: DataLoader) -> np.ndarray:
    folds = len(result.models)
    preds_proba = np.zeros((test_dl.dataset.x.shape[0],))
    for model in tqdm(result.models.values(), total=folds):
        preds_proba += (
            model.predict(test_dl) / folds
        )
    assert len(preds_proba) == len(test_dl.dataset.x)
    return preds_proba

def inference_dae(result: ModelResult,
                  train_cont: DataContainer) -> DataContainer:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dl = train_cont.get_dl_dataloader_for_testing(batch_size=512,
                                                       num_workers=64)
    model = list(result.models.values())[0]

    torch_test = []
    for i, x in enumerate(test_dl):
        x = x.to(device)
        with torch.no_grad():
            x = model.feature(x)
        torch_test.append(x)
    torch_test = torch.cat(torch_test, dim=0)

    new_test_cont = DataContainer(df=pd.DataFrame(torch_test.detach().cpu().numpy()),
                                  df_y=train_cont.df_y,
                                  len_cat=0,
                                  len_num=torch_test.shape[1])
    return new_test_cont





