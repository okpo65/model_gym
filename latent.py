import logging
import os
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.dataset.dataset import load_train_data, load_test_data
from src.models.infer import inference, load_model, inference_mlp, inference_dae, inference_feature_importance, \
    inference_dae_gmm, inference_dae_mlp, inference_tabnet, inference_tabnet_latent
from src.evaluation.evaluation import css_metric
from src.dataset.preprocessing import PreprocessorApplicator, get_preprocessor_path
from src.utils.constants import representation_key, __all_model__


@hydra.main(config_path="config/", config_name='latent', version_base='1.2.0')
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

    test_cont = preprocessor.perform(X_test, y_test)

    device = torch.device(cfg.dataset.device)
    # using representation learning features
    if representation_key in cfg.keys():
        model_path = Path(get_original_cwd()) / cfg.representation.path / cfg.representation.result
        # model load
        dae_results = load_model(model_path)
        test_cont = inference_dae_gmm(dae_results, test_cont, device)

    # infer test dataset
    model_name = cfg.model.name
    if model_name == __all_model__.bottleneck_dae:
        latent = inference_dae(results, test_cont, device)
    elif model_name == __all_model__.dae_mlp:
        test_dl = test_cont.get_test_dataloader(batch_size=cfg.model.batch_size,
                                                num_workers=cfg.dataset.num_workers)
        latent = inference_dae_mlp(results, test_dl, device)
    elif model_name == __all_model__.tabnet:
        X_test, y_test = test_cont.get_dataframe()
        latent = inference_tabnet_latent(results, X_test, y_test)

    latent_path = (
            Path(get_original_cwd()) / cfg.output.path / cfg.output.name
    )
    df_latent, _ = latent.get_dataframe()
    df_latent.columns = [f"{c}" for c in df_latent.columns.tolist()]
    df_latent[cfg.dataset.target_name] = y_test.values
    df_latent.to_parquet(latent_path)

if __name__ == "__main__":
    _main()

