import os
from pathlib import Path
import torch
from hydra.utils import get_original_cwd
from src.dataset.dataset import load_train_data, DataContainer
from src.models.boosting import LGBMTrainer, CatBoostTrainer, XGBoostTrainer
from src.models.tabnet import TabNetTrainer, TabNetPretrainModel
from src.models.autoencoder import DAE
from src.models.mlp import MLP
from src.models.dae_mlp import DAEMLP
from src.models.gmm_dae import GMMDAE
from src.evaluation.evaluation import css_metric
from src.utils.utils import WANDB_KEY
from src.dataset.preprocessing import PreprocessorApplicator, get_preprocessor_path
import hydra
from src.utils.utils import DictX
from omegaconf import DictConfig
from src.models.infer import inference_dae, load_model
import wandb
from src.utils.constants import representation_key, __all_model__


@hydra.main(config_path='config/', config_name='main', version_base='1.2.0')
def _main(cfg: DictConfig):
    """
    :param cfg: hydra configuration

    Control the overall course of Learning
    """

    # wandb login
    wandb.login(key=WANDB_KEY)

    # load Data
    X_train, y_train = load_train_data(cfg)

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

    train_cont = preprocessor.perform(X_train, y_train)

    print('-------------\n Started Training \n-----------', train_cont.get_dataframe())
    device = torch.device(cfg.dataset.device)
    # using Representation Learning Features
    if representation_key in cfg.keys():
        model_path = Path(get_original_cwd()) / cfg.representation.path / cfg.representation.result
        # model load
        results = load_model(model_path)
        train_cont = inference_dae(results, train_cont, device)

    # model training
    torch.set_num_threads(cfg.dataset.num_workers)
    model_name = cfg.model.name
    model = get_model(model_name, cfg)

    model.train(train_cont, device)

    # model save
    model.save_model()
    wandb.finish()


def get_model(model_name, cfg):
    """
    :param model_name: model name from hydra config
    :return: Model Object

    Determine the model object with the model name in the config file
    """
    if model_name == __all_model__.lgbm:
        model = LGBMTrainer(config=cfg, metric=css_metric)
    elif model_name == __all_model__.xgboost:
        model = XGBoostTrainer(config=cfg, metric=css_metric)
    elif model_name == __all_model__.catboost:
        model = CatBoostTrainer(config=cfg, metric=css_metric)
    elif model_name == __all_model__.mlp:
        model = MLP(config=cfg, metric=css_metric)
    elif model_name == __all_model__.dae_mlp:
        model = DAEMLP(config=cfg, metric=css_metric)
    elif model_name == __all_model__.deepstack_dae or model_name == __all_model__.bottleneck_dae or model_name == __all_model__.transformer_dae:
        model = DAE(config=cfg)
    elif model_name == __all_model__.gmm_dae:
        model = GMMDAE(config=cfg)
    elif model_name == __all_model__.tabnet:
        model = TabNetTrainer(config=cfg, metric=css_metric)
    elif model_name == __all_model__.tabnet_pretrainer:
        model = TabNetPretrainModel(config=cfg)
    return model


if __name__ == "__main__":
    _main()
