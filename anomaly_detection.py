import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.dataset.dataset import load_train_data, load_test_data
from src.models.infer import inference, load_model, inference_mlp, inference_dae, inference_dae_reconstruction
from src.evaluation.evaluation import css_metric, reconstruction_confidence
from src.dataset.preprocessing import Preprocessor
from src.utils.utils import DictX
# from utils.utils import seed_everything

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

@hydra.main(config_path="config/", config_name='anomaly_detection', version_base='1.2.0')
def _main(cfg: DictConfig):

    path = Path(get_original_cwd())/ cfg.model.path / cfg.model.result

    # model load
    results = load_model(cfg, path)
    # data load
    X_train, y_train = load_train_data(cfg)
    X_test, y_test = load_test_data(cfg)
    print("results!", results)
    # preprocessing
    cat_features = [*cfg.features.cat_features]
    num_features = sorted(list(set(X_train.columns.tolist()) - set(cat_features)))
    preprocessor = Preprocessor(cfg.preprocessing,
                                X_train,
                                y_train,
                                X_test,
                                num_features=num_features,
                                cat_features=cat_features)

    train_cont, test_cont = preprocessor.perform()

    model_path = Path(get_original_cwd()) / cfg.model.path / cfg.model.result
    # model load
    dae_results = load_model(cfg, model_path)
    df_before, _ = test_cont.get_dataframe()
    df_before.columns = [f"{c}_cat" for c in df_before.columns.tolist()[:test_cont.len_cat]] + [f"{c}" for c in num_features]
    df_before.to_parquet(cfg.output.before_path)
    # dae reconstruction
    test_cont = inference_dae_reconstruction(dae_results, test_cont)
    df_after, _ = test_cont.get_dataframe()
    df_after.columns = [f"{c}_cat" for c in df_after.columns.tolist()[:test_cont.len_cat]] + [f"{c}" for c in num_features]
    df_after.to_parquet(cfg.output.after_path)

    df_confidence = reconstruction_confidence(df_before,
                                              df_after,
                                              cat_features=cat_features,
                                              num_features=num_features)
    df_confidence.to_parquet('../CSS/df_confidence.parquet')

if __name__ == "__main__":
    _main()

