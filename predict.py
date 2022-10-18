import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.dataset.dataset import load_test_data
from src.models.infer import inference, load_model
from src.evaluation.evaluation import css_metric
# from utils.utils import seed_everything

@hydra.main(config_path="config/", config_name='predict', version_base='1.2.0')
def _main(cfg: DictConfig):
    path = Path(get_original_cwd())/ cfg.output.path

    # model load
    results = load_model(cfg, cfg.model.result)
    # data load
    X_test, y_test = load_test_data(cfg)
    # infer test
    preds = inference(results, X_test)
    if y_test is not None:
        print(f"KS: {css_metric(y_test, preds)}")

if __name__ == "__main__":
    _main()

