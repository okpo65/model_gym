import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.dataset.dataset import load_train_data, load_test_data
from src.models.infer import inference, load_model, inference_mlp, inference_dae
from src.evaluation.evaluation import css_metric
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

@hydra.main(config_path="config/", config_name='predict', version_base='1.2.0')
def _main(cfg: DictConfig):

    path = Path(get_original_cwd())/ cfg.model.path / cfg.model.result
    model_name = cfg.model.name
    # model load
    results = load_model(cfg, path)
    # data load
    X_train, y_train = load_train_data(cfg)
    X_test, y_test = load_test_data(cfg)
    print("results!", results)
    # preprocessing
    cat_features = [*cfg.features.cat_features]
    preprocessor = Preprocessor(cfg.preprocessing,
                                X_train,
                                y_train,
                                X_test,
                                cat_features=cat_features)

    train_cont, test_cont = preprocessor.perform()
    # representation learning
    if 'representation' in cfg.keys():
        model_path = Path(get_original_cwd()) / cfg.representation.path / cfg.representation.result
        # model load
        dae_results = load_model(cfg, model_path)
        test_cont = inference_dae(dae_results, test_cont)
    # infer test
    if model_name == __all_model__.mlp:
        test_dl = test_cont.get_dl_dataloader_for_testing(batch_size=cfg.model.batch_size,
                                                          num_workers=cfg.model.num_workers)
        preds = inference_mlp(results, test_dl)
    else:
        X_test, _ = test_cont.get_dataframe()
        preds = inference(results, X_test)

    # 결과 저장
    model_path = (
        Path(get_original_cwd()) / cfg.output.path / cfg.output.name
    )
    pd.DataFrame(preds).to_csv(model_path)

    print(f"KS: {css_metric(preds, y_test)}")

if __name__ == "__main__":
    _main()

