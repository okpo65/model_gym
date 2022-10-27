import logging
import warnings
from pathlib import Path
from typing import Tuple
import torch
import pandas as pd
from hydra.utils import get_original_cwd
from src.dataset.dataset import load_train_data, load_test_data, DataContainer
from src.models.boosting import LGBMTrainer, CatBoostTrainer, XGBoostTrainer
from src.models.tabnet import TabNetTrainer
from src.models.autoencoder import DAE
from src.models.mlp import MLP
from src.evaluation.evaluation import css_metric
from src.utils.utils import WANDB_KEY
from src.dataset.preprocessing import Preprocessor
import hydra
from src.utils.utils import DictX
from omegaconf import DictConfig
from src.models.infer import inference_dae, load_model
import wandb

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
    # wandb
    wandb.login(key=WANDB_KEY)
    # wandb.init(project='model_gym',
    #            name=cfg.model.result,
    #            reinit=True)
    # get dataset
    X_train, y_train = load_train_data(cfg)

    # preprocessing
    cat_features = [*cfg.features.cat_features]
    num_features = sorted(list(set(X_train.columns.tolist()) - set(cat_features)))
    preprocessor = Preprocessor(cfg.preprocessing,
                                X_train,
                                y_train,
                                num_features=num_features,
                                cat_features=cat_features)
    train_cont, _ = preprocessor.perform()
    print('-------------\n Started Training \n-----------', train_cont.get_dataframe())
    # representation learning
    if 'representation' in cfg.keys():
        model_path = Path(get_original_cwd()) / cfg.representation.path / cfg.representation.result
        # model load
        results = load_model(cfg, model_path)
        train_cont = inference_dae(results, train_cont)
    # model training
    torch.set_num_threads(cfg.dataset.num_workers)
    model_name = cfg.model.name
    print('model_name: ', model_name)
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
    wandb.finish()
if __name__ == "__main__":
    _main()
