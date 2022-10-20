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
from src.evaluation.evaluation import css_metric
from src.dataset.preprocessing import Preprocessor
import hydra
from omegaconf import DictConfig

@hydra.main(config_path='config/', config_name='main', version_base='1.2.0')
def _main(cfg: DictConfig):
    X_train, y_train = load_train_data(cfg)

    preprocessor = Preprocessor(cfg.preprocessing,
                                X_train,
                                cat_features=[*cfg.features.cat_features])
    train_cont, _ = preprocessor.perform()
    df_tmp, _ = train_cont.get_dataframe()
    print("sdfsdf",df_tmp)
    dae = DAE(config=cfg)
    dae.train(train_cont)
    dae.save_model()

    # tabnet = TabNetTrainer(params=None,
    #                        config=cfg,
    #                        metric=css_metric)
    # tabnet.train(X_train, y_train)
    # tabnet.save_model()

    # dae = DAE(config=cfg, metric=css_metric)
    # dae.train(X_train)
    # dae.save_model()


if __name__ == "__main__":
    _main()
