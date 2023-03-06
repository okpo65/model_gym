import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.dataset.dataset import load_train_data, load_test_data
from src.models.infer import load_model
from src.evaluation.evaluation import reconstruction_confidence
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

@hydra.main(config_path="config/", config_name='gap_bridge', version_base='1.2.0')
def _main(cfg: DictConfig):
    """
    Analyze confidence level after reconstructing the data with a model trained with DAE
    """

    # model load
    path = Path(get_original_cwd()) / cfg.model.path / cfg.model.result
    results = load_model(cfg, path)

    # data load
    X_train, y_train = load_train_data(cfg)
    X_test, y_test = load_test_data(cfg)
    print("results!", results)

    # preprocessing
    cat_features = [*cfg.features.cat_features] if 'cat_features' in cfg.features.keys() else []
    num_features = sorted(list(set(X_train.columns.tolist()) - set(cat_features)))
    preprocessor = Preprocessor(cfg.preprocessing,
                                X_train,
                                y_train,
                                X_test,
                                num_features=num_features,
                                cat_features=cat_features)

    train_cont, test_cont = preprocessor.perform()

    model_name = cfg.model.name
    model = get_model(model_name, cfg)

    model.train(train_cont, device)

    # model save
    model.save_model()
    wandb.finish()

if __name__ == "__main__":
    _main()

