import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import tqdm
from src.dataset.dataset import load_train_data, load_test_data
from src.models.infer import load_model, inference_shap, inference_dae, inference_shap_v2
from src.dataset.preprocessing import Preprocessor
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

@hydra.main(config_path="config/", config_name='shap', version_base='1.2.0')
def _main(cfg: DictConfig):

    """
    :param cfg: predict config file

    Explain model with shapley value
    """
    # model load
    path = Path(get_original_cwd()) / cfg.model.path / cfg.model.result
    results = load_model(cfg, path)

    # data load
    X_train, y_train = load_train_data(cfg)
    X_test, y_test = load_test_data(cfg)
    print("results!", results)

    # preprocessing as same as for training
    cat_features = [*cfg.features.cat_features] if 'cat_features' in cfg.features.keys() else []
    num_features = sorted(list(set(X_train.columns.tolist()) - set(cat_features)))
    preprocessor = Preprocessor(cfg.preprocessing,
                                X_train,
                                y_train,
                                X_test,
                                num_features=num_features,
                                cat_features=cat_features)

    train_cont, test_cont = preprocessor.perform()

    # using representation learning features
    if representation_key in cfg.keys():
        model_path = Path(get_original_cwd()) / cfg.representation.path / cfg.representation.result
        # model load
        dae_results = load_model(cfg, model_path)
        test_cont = inference_dae(dae_results, test_cont)

    # get shapley value list
    shap_value_list = inference_shap_v2(results, test_cont)
    shap_value_list.to_parquet('df_ranking_test.parquet')
    # for shap in shap_value_list:
    #     print(shap, len(shap[0]), len(shap[1]))
    #     print(len(shap[0][0]), len(shap[1][0]))

if __name__ == "__main__":
    _main()

