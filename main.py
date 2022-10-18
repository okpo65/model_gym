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
import hydra
from omegaconf import DictConfig

@hydra.main(config_path='config/', config_name='main', version_base='1.2.0')
def _main(cfg: DictConfig):
    X_train, y_train = load_train_data(cfg)

    # tabnet = TabNetTrainer(params=None,
    #                        config=cfg,
    #                        metric=css_metric)
    # tabnet.train(X_train, y_train)
    # tabnet.save_model()

    dae = DAE(config=cfg, metric=css_metric)
    X, len_cat, len_num = get_dataset_with_cat_handler(X_train,
                                                       [*self.config.features.cat_features])
    dae_dl = get_dae_dataset(X,
                             batch_size=self.config.model.data_batch_size,
                             num_workers=self.config.model.num_workers)
    dae.train(X_train)
    dae.save_model()


if __name__ == "__main__":
    _main()


