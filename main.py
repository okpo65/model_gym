import logging
import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd
from hydra.utils import get_original_cwd
from src.dataset.dataset import load_train_data, load_test_data
from src.models.boosting import LGBMTrainer, CatBoostTrainer, XGBoostTrainer
from src.models.tabnet import TabNetTrainer
from src.models.autoencoder import DAE
from src.models.mlp import MLP
from src.evaluation.evaluation import css_metric
from src.dataset.preprocessing import Preprocessor
import hydra
from src.utils.utils import DictX
from omegaconf import DictConfig


__all_model__ = DictX(
    catboost='catboost',
    lgbm='lgbm',
    xgboost='xgboost',
    mlp='mlp',
    deepstack_dae='deepstack_dae',
    bottleneck_dae='bottleneck_dae',
    transformer_dae='transformer_dae',
    tabnet='tabnet'
)
@hydra.main(config_path='config/', config_name='main', version_base='1.2.0')
def _main(cfg: DictConfig):
    # get dataset
    X_train, y_train = load_train_data(cfg)

    # preprocessing
    preprocessor = Preprocessor(cfg.preprocessing,
                                X_train,
                                y_train,
                                cat_features=[*cfg.features.cat_features])

    train_cont, _ = preprocessor.perform()

    # model training
    model_name = cfg.model.name
    if model_name == __all_model__.lgbm:
        model = LGBMTrainer(config=cfg, metric=css_metric)
    elif model_name == __all_model__.xgboost:
        model = XGBoostTrainer(config=cfg, metric=css_metric)
    elif model_name == __all_model__.catboost:
        model = CatBoostTrainer(config=cfg, metric=css_metric)
    elif model_name == __all_model__.mlp:
        model = MLP(config=cfg, metric=css_metric)
    elif model_name == __all_model__.deepstack_dae or model_name == __all_model__.bottleneck_dae or model_name == __all_model__.transformer_dae:
        model = DAE(config=cfg)
    elif model_name == __all_model__.tabnet:
        model = TabNetTrainer(config=cfg, metric=css_metric)

    model.train(train_cont)
    model.save_model()

if __name__ == "__main__":
    _main()
