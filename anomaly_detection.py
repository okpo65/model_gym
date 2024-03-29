import logging
import os
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.dataset.dataset import load_train_data, load_test_data
from src.dataset.preprocessing import PreprocessorApplicator, get_preprocessor_path
from src.models.infer import load_model, inference_dae_reconstruction, inference_dae
from src.evaluation.evaluation import reconstruction_confidence


@hydra.main(config_path="config/", config_name='anomaly_detection', version_base='1.2.0')
def _main(cfg: DictConfig):
    """
    Analyze confidence level after reconstructing the data with a model trained with DAE
    """

    # data load
    X_train, y_train = load_train_data(cfg)
    X_test, y_test = load_test_data(cfg)

    # preprocessing
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

    # model load
    model_path = Path(get_original_cwd()) / cfg.model.path / cfg.model.result
    dae_results = load_model(model_path)
    df_before, _ = test_cont.get_dataframe()
    df_before.columns = [f"{c}_cat" for c in df_before.columns.tolist()[:test_cont.len_cat]] + [f"{c}" for c in num_features]
    df_before.to_parquet(cfg.output.before_path)

    # dae reconstruction
    reconstruted_test_cont = inference_dae_reconstruction(dae_results, test_cont, device)
    df_after, _ = reconstruted_test_cont.get_dataframe()
    df_after.columns = [f"{c}_cat" for c in df_after.columns.tolist()[:reconstruted_test_cont.len_cat]] + [f"{c}" for c in num_features]
    df_after.to_parquet(cfg.output.after_path)

    # dae representation
    representation_test_cont = inference_dae(dae_results, test_cont, device)
    df_representation, _ = representation_test_cont.get_dataframe()
    df_representation.columns = [f"{c}" for c in df_representation.columns]
    df_representation.to_parquet(cfg.output.representation_path)

    # get confidence level from each cell
    df_confidence = reconstruction_confidence(df_before,
                                              df_after,
                                              cat_features=cat_features,
                                              num_features=num_features)
    df_confidence.to_parquet(cfg.output.confidence_path)

if __name__ == "__main__":
    _main()

