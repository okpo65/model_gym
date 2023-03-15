import argparse
import logging
from pathlib import Path

import hydra
import pandas as pd
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.dataset.dataset import load_train_data, load_test_data
from src.models.infer import inference, load_model, inference_mlp, inference_dae, inference_feature_importance
from src.evaluation.evaluation import css_metric
from src.dataset.preprocessing import Preprocessor, Preprocessor2
from src.utils.constants import JARVIS_NULL_REPLACEMENTS
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

@hydra.main(config_path="config/", config_name='onpremise', version_base='1.2.0')
def _main(cfg: DictConfig):

    """
    :param cfg: predict config file

    Evaluate model performance with already trained model
    """
    # model load
    root_path = cfg.dataset.root # '/home/peoplefund/workspace/jihwan-css/ModelStorage'# cfg.dataset.root # Path(get_original_cwd())
    dae_path = root_path + '/res/models/' + 'deepstack_dae_4500_standard_jb_no_cat'
    lgbm_path = root_path + '/res/models/' + 'lgbm_dae_standard_jb_v2_no_cat'
    mlp_path = root_path + '/res/models/' + 'mlp_4500dae_standard_jb_v2_no_cat'
    
    dae_model = load_model(dae_path)
    lgbm_model = load_model(lgbm_path)
    mlp_model = load_model(mlp_path)

    # data load
    # X_train, y_train = load_train_data(cfg)
    X_test, y_test = load_test_data(cfg)
    print("load data")
    # preprocessing as same as for training
    num_features = [*cfg.features.total_features]
    preprocessor = Preprocessor2(X_test,
                                 num_features=num_features)

    test_cont = preprocessor.perform()
    device = torch.device('cpu')

    # dae inference
    test_cont = inference_dae(dae_model, test_cont, device)

    # mlp inference
    test_dl = test_cont.get_test_dataloader(batch_size=64,
                                            num_workers=3)

    mlp_preds = inference_mlp(mlp_model, test_dl, device)

    # lgbm inference
    # X_test, _ = test_cont.get_dataframe()
    # lgbm_preds = inference(lgbm_model, X_test)

    # save predidtion
    X_test = pd.read_csv(cfg.dataset.test)
    X_test['css_model_B_proba'] = mlp_preds * 1 # + lgbm_preds * 0.0
    X_test['css_model_B_score'] = np.log((1-X_test['css_model_B_proba'])/X_test['css_model_B_proba']) / np.log(2)*40+500

    X_test.to_csv(cfg.dataset.save_path)

if __name__ == "__main__":
    _main()

