import logging
import os
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import tqdm
from src.dataset.dataset import load_train_data, load_test_data
from src.models.infer import load_model, inference_uncertainty,inference_uncertainty_2, inference_dae
from src.dataset.preprocessing import Preprocessor, get_preprocessor_path, PreprocessorApplicator
from src.utils.utils import DictX

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
representation_key = 'representation'

@hydra.main(config_path="config/", config_name='uncertainty', version_base='1.2.0')
def _main(cfg: DictConfig):

    """
    :param cfg: predict config file

    Uncertainty of model output
    """
    # model load
    path = Path(get_original_cwd()) / cfg.model.path / cfg.model.result
    results = load_model(cfg, path)

    # data load
    X_train, y_train = load_train_data(cfg)
    # X_test, y_test = load_test_data(cfg)
    print("results!", results)

    cat_features = [*cfg.features.cat_features] if 'cat_features' in cfg.features.keys() else []
    num_features = sorted(list(set(X_train.columns.tolist()) - set(cat_features)))

    preprocessor_path = get_preprocessor_path(cfg)
    if not os.path.exists(preprocessor_path):
        os.makedirs(preprocessor_path)

    preprocessor = PreprocessorApplicator(cfg.preprocessing,
                                          X_train,
                                          y_train,
                                          num_features=num_features,
                                          cat_features=cat_features,
                                          preprocessor_path=preprocessor_path)
    if not os.listdir(preprocessor_path) or cfg.preprocessor_applicator.refresh == True:
        preprocessor.save()

    train_cont = preprocessor.perform(X_train, y_train)

    print('-------------\n Started Training \n-----------', train_cont.get_dataframe())
    device = torch.device(cfg.dataset.device)
    # using Representation Learning Features
    if representation_key in cfg.keys():
        model_path = Path(get_original_cwd()) / cfg.representation.path / cfg.representation.result
        # model load
        results = load_model(model_path)
        train_cont = inference_dae(results, train_cont, device)

    # get shapley value list
    # test_dl = test_cont.get_test_dataloader(cfg.model.batch_size, cfg.dataset.num_workers)
    df = inference_uncertainty_2(results,
                                 X_train.iloc[:1],
                                 n_process=300)

    df.to_parquet('df_mlp_transformer_dae_uncertainty_2.parquet')

if __name__ == "__main__":
    _main()

