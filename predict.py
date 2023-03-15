import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.dataset.dataset import load_train_data, load_test_data
from src.models.infer import inference, load_model, inference_mlp, inference_dae, inference_feature_importance
from src.evaluation.evaluation import css_metric
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

@hydra.main(config_path="config/", config_name='predict', version_base='1.2.0')
def _main(cfg: DictConfig):

    """
    :param cfg: predict config file

    Evaluate model performance with already trained model
    """
    # model load
    path = Path(get_original_cwd()) / cfg.model.path / cfg.model.result
    results = load_model(path)

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
    # test_cont.get_dataframe()[0].to_parquet('../CSS/dataset/test/df_dae_03test_2_proxy_processed.parquet')

    device = torch.device(cfg.dataset.device)
    # using representation learning features
    if representation_key in cfg.keys():
        model_path = Path(get_original_cwd()) / cfg.representation.path / cfg.representation.result
        # model load
        dae_results = load_model(model_path)
        test_cont = inference_dae(dae_results, test_cont, device)

    # infer test dataset
    model_name = cfg.model.name
    if model_name == __all_model__.mlp:
        test_dl = test_cont.get_test_dataloader(batch_size=cfg.model.batch_size,
                                                num_workers=cfg.dataset.num_workers)
        preds = inference_mlp(results, test_dl, device)
    else:
        X_test, _ = test_cont.get_dataframe()
        preds = inference(results, X_test)

    # save predidtion
    model_path = (
        Path(get_original_cwd()) / cfg.output.path / cfg.output.name
    )
    pd.DataFrame(preds).to_csv(model_path)

    print(f"KS: {css_metric(preds, y_test)}")

if __name__ == "__main__":
    _main()

