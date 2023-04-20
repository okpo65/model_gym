import io
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from ..utils.utils import SwapNoiseMasker
import torch
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
from ..dataset.dataset import DataContainer
from .base_model import ModelResult


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_model(model_path: str) -> ModelResult:
    """
    :param config: Hydra config
    :param model_name: model_name
    :return: ModelResult Object
    """

    with open(model_path, "rb") as output:
        model_result = pickle.load(output)

    return model_result

def load_pretrained_model(model_path: str):
    result = load_model(model_path)
    return result.models.values()[0]

def inference(result: ModelResult,
              X_test: pd.DataFrame) -> np.ndarray:
    """
    :param result: ModelResult Object
    :param X_test: dataframe
    :return: predict probabilities for each class

    inference for boosting model
    """

    folds = len(result.models)
    preds_proba = np.zeros((X_test.shape[0], ))

    for model in tqdm(result.models.values(), total=folds):
        preds_proba += (
            model.predict(X_test) / folds
            if isinstance(model, lgb.Booster)
            else model.predict(xgb.DMatrix(X_test)) / folds
            if isinstance(model, xgb.Booster)
            else model.predict_proba(X_test.to_numpy())[:, 1] / folds
            if isinstance(model, CatBoostClassifier)
            else model.predict(X_test.to_numpy())
        )
    assert len(preds_proba) == len(X_test)

    return preds_proba

def inference_tabnet(result: ModelResult,
                     X_test: pd.DataFrame) -> np.ndarray:
    """
    :param result: ModelResult Object
    :param X_test: dataframe
    :return: predict probabilities for each class

    inference for boosting model
    """

    folds = len(result.models)
    preds_proba = np.zeros((X_test.shape[0], ))

    for model in tqdm(result.models.values(), total=folds):
        preds_proba += (
            model.predict_proba(X_test.to_numpy())[:, 1] / folds
        )
    assert len(preds_proba) == len(X_test)

    return preds_proba

def inference_tabnet_latent(result: ModelResult,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            num_workers: int) -> list[np.ndarray]:
    """
    :param result: ModelResult Object
    :param X_test: dataframe
    :return: predict probabilities for each class

    inference latent for boosting model
    """

    folds = len(result.models)

    for model in tqdm(result.models.values(), total=folds):
        latent_list = model.predict_latent(X_test.to_numpy(), num_workers=num_workers)

    assert latent_list[0].shape[0] == len(X_test)

    test_cont_list = []
    for latent in latent_list:
        new_test_cont = DataContainer(df=pd.DataFrame(latent),
                                      df_y=y_test,
                                      len_cat=0,
                                      len_num=latent.shape[1])
        test_cont_list.append(new_test_cont)
    return test_cont_list


def inference_dae_mlp(result: ModelResult,
                      test_dl: DataLoader,
                      device: torch.device) -> np.ndarray:
    """
    :param result: ModelResult Object
    :param X_test: dataframe
    :return: predict probabilities for each class

    inference for boosting model
    """
    folds = len(result.models)
    preds_proba = np.zeros((test_dl.dataset.x.shape[0],))
    for model in tqdm(result.models.values(), total=folds):
        _model = model.to(device)
        _model.eval()
        predictions = []
        with torch.no_grad():
            prediction = _model.predict(test_dl)
        predictions.append(prediction)
        predictions = np.concatenate(predictions).reshape(-1)
        preds_proba += (
                predictions / folds
        )
    assert len(preds_proba) == len(test_dl.dataset.x)
    return preds_proba

def inference_feature_importance(result: ModelResult, cat_features, num_features) -> pd.DataFrame:
    folds = len(result.models)
    df_fi = pd.DataFrame()
    df_fi['feature_names'] = cat_features + sorted(num_features)
    for fold, model in tqdm(enumerate(result.models.values()), total=folds):
        feature_importance = np.array(model.feature_importances_)
        df_fi[f'feature_importance_{fold}'] = feature_importance

    return df_fi

def inference_mlp(result: ModelResult,
                  test_dl: DataLoader,
                  device: torch.device) -> np.ndarray:
    """
    :param result: ModelResult Object
    :param test_dl: test dataloader
    :return: predict probabilities for each class

    inference for mlp model
    """
    folds = len(result.models)
    preds_proba = np.zeros((test_dl.dataset.x.shape[0],))
    for model in tqdm(result.models.values(), total=folds):
        _model = model.to(device)
        _model.eval()
        predictions = []
        with torch.no_grad():
            for i, x in enumerate(test_dl):
                if isinstance(x, list):
                    x = x[0]
                x = x.to(device)
                prediction = _model.forward(x)
                predictions.append(prediction.detach().cpu().numpy())
        predictions = np.concatenate(predictions).reshape(-1)
        preds_proba += (
            predictions / folds
        )
    assert len(preds_proba) == len(test_dl.dataset.x)
    return preds_proba

def inference_dae(result: ModelResult,
                  train_cont: DataContainer,
                  device: torch.device) -> DataContainer:
    """
    :param result: ModelResult Object
    :param train_cont: DataContainer to be DAE representation features
    :return: new DataContainer with representation features
    """
    test_dl = train_cont.get_test_dataloader(batch_size=512,
                                             num_workers=64)
    model = list(result.models.values())[0]
    model.eval()
    model = model.to(device)
    torch_test = []
    for i, x in enumerate(test_dl):
        x = x.to(device)
        with torch.no_grad():
            x = model.feature(x)
        torch_test.append(x)
    torch_test = torch.cat(torch_test, dim=0)

    new_test_cont = DataContainer(df=pd.DataFrame(torch_test.detach().cpu().numpy()),
                                  df_y=train_cont.df_y,
                                  len_cat=0,
                                  len_num=torch_test.shape[1])
    return new_test_cont


def inference_dae_gmm(result: ModelResult,
                      train_cont: DataContainer,
                      device: torch.device) -> DataContainer:
    """
    :param result: ModelResult Object
    :param train_cont: DataContainer to be DAE representation features
    :return: new DataContainer with representation features
    """
    test_dl = train_cont.get_test_dataloader(batch_size=512,
                                             num_workers=64)
    model = list(result.models.values())[0]
    model = model.to(device)
    model.eval()
    torch_test = []
    for i, x in enumerate(test_dl):
        x = x.to(device)
        with torch.no_grad():
            enc, dec, z, gamma = model.forward_gamma(x)
        torch_test.append(z)
    torch_test = torch.cat(torch_test, dim=0)

    new_test_cont = DataContainer(df=pd.DataFrame(torch_test.detach().cpu().numpy()),
                                  df_y=train_cont.df_y,
                                  len_cat=0,
                                  len_num=torch_test.shape[1])
    return new_test_cont

def inference_dae_gmm_gamma(result: ModelResult,
                      train_cont: DataContainer,
                      device: torch.device) -> pd.DataFrame:
    """
    :param result: ModelResult Object
    :param train_cont: DataContainer to be DAE representation features
    :return: new DataContainer with representation features
    """
    test_dl = train_cont.get_test_dataloader(batch_size=512,
                                             num_workers=64)
    model = list(result.models.values())[0]
    model = model.to(device)
    model.eval()
    torch_test = []
    for i, x in enumerate(test_dl):
        x = x.to(device)
        with torch.no_grad():
            enc, dec, z, gamma = model.forward_gamma(x)
        torch_test.append(gamma)
    torch_test = torch.cat(torch_test, dim=0)
    df_gamma = pd.DataFrame(torch_test.detach().cpu().numpy())
    return df_gamma


def inference_dae_reconstruction(result: ModelResult,
                                 train_cont: DataContainer,
                                 device: torch.device) -> DataContainer:
    """
    :param result: ModelResult Object
    :param train_cont: dataloader to be DAE reconstructed data
    :return: new dataContainer with reconstructed data
    """
    test_dl = train_cont.get_test_dataloader(batch_size=512,
                                             num_workers=64)
    model = list(result.models.values())[0]
    model = model.to(device)
    model.eval()
    torch_test = []
    for i, x in enumerate(test_dl):
        x = x.to(device)
        with torch.no_grad():
            x = model.reconstructed_feature(x)
        torch_test.append(x)
    torch_test = torch.cat(torch_test, dim=0)

    new_test_cont = DataContainer(df=pd.DataFrame(torch_test.detach().cpu().numpy()),
                                  df_y=train_cont.df_y,
                                  len_cat=train_cont.len_cat,
                                  len_num=train_cont.len_num)
    return new_test_cont


def shap_feature_ranking(df, shap_values, columns=[]):
    if not columns: columns = df.columns.tolist()  # If columns are not given, take all columns

    c_idxs = []
    for column in columns: c_idxs.append(df.columns.get_loc(column))  # Get column locations for desired columns in given dataframe
    if isinstance(shap_values, list):  # If shap values is a list of arrays (i.e., several classes)
        means = [np.abs(shap_values[class_][:, c_idxs]).mean(axis=0) for class_ in range(len(shap_values))]  # Compute mean shap values per class
        shap_means = np.sum(np.column_stack(means), 1)  # Sum of shap values over all classes
    else:  # Else there is only one 2D array of shap values
        assert len(shap_values.shape) == 2, 'Expected two-dimensional shap values array.'
        shap_means = np.abs(shap_values).mean(axis=0)


    # Put into dataframe along with columns and sort by shap_means, reset index to get ranking
    df_ranking = pd.DataFrame({'feature': columns, 'mean_shap_value': shap_means}).sort_values(by='mean_shap_value', ascending=False).reset_index(drop=True)
    df_ranking.index += 1
    return df_ranking

def shap_feature_ranking_2(df, shap_values, columns=[]):
    if not columns: columns = df.columns.tolist()  # If columns are not given, take all columns
    df_ranking = pd.DataFrame(data=shap_values, columns=columns)
    return df_ranking

def inference_shap(result: ModelResult,
                   test_cont: DataContainer,
                   device: torch.device):
    folds = len(result.models)
    shap_value_list = []
    for model in tqdm(result.models.values(), total=folds):
        X_test = test_cont.get_dataframe()[0].iloc[:200]
        if isinstance(model, lgb.Booster) or isinstance(model, xgb.Booster) or isinstance(model, CatBoostClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)[1]
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]

        else:
            X_test_torch = torch.Tensor(X_test.values).to(device)
            explainer = shap.DeepExplainer(model, X_test_torch)
            shap_values = explainer.shap_values(X_test_torch)
            shap_values = np.array(shap_values)
        shap_value_list.append(shap_values)
        shap.summary_plot(shap_values, X_test, plot_type='bar')
        plt.savefig('shap_summary.png')
        df_ranking = shap_feature_ranking_2(X_test, shap_values)
    return df_ranking

def inference_shap_v2(result: ModelResult,
                   test_cont: DataContainer):
    folds = len(result.models)

    def predict_proba(X_test):
        preds_proba = np.zeros((X_test.shape[0],))
        for model in tqdm(result.models.values(), total=folds):
            if isinstance(model, lgb.Booster) or isinstance(model, xgb.Booster) or isinstance(model,
                                                                                              CatBoostClassifier):
                preds_proba += (
                    model.predict(X_test) / folds
                    if isinstance(model, lgb.Booster)
                    else model.predict(xgb.DMatrix(X_test)) / folds
                    if isinstance(model, xgb.Booster)
                    else model.predict_proba(X_test.to_numpy())[:, 1] / folds
                    if isinstance(model, CatBoostClassifier)
                    else model.predict(X_test.to_numpy())
                )
        return preds_proba

    X_test = test_cont.get_dataframe()[0].iloc[:100]
    print("X_test!!", X_test)
    explainer = shap.KernelExplainer(predict_proba, X_test, link='logit', nsamples=100)
    shap_values = explainer.shap_values(X_test)
    df_ranking = shap_feature_ranking_2(X_test, shap_values)
    return df_ranking

def inference_uncertainty(result: ModelResult,
                          test_dl: DataLoader,
                          n_process: int) -> pd.DataFrame:
    folds = len(result.models)
    df = pd.DataFrame()
    for fold, model in tqdm(enumerate(result.models.values()), total=folds):
        mean, std = model.predict_uncertainty(test_dl,
                                              n_process=n_process)
        df[f"mean_{fold}"] = mean
        df[f"std_{fold}"] = std

    return df


def inference_uncertainty_2(result: ModelResult,
                          X_test: pd.DataFrame,
                          n_process: int) -> pd.DataFrame:
    folds = len(result.models)
    df = pd.DataFrame()
    for fold, model in tqdm(enumerate(result.models.values()), total=folds):
        mean, std = model.predict_uncertainty(X_test,
                                              n_process=n_process)
        df[f"mean_{fold}"] = mean
        df[f"std_{fold}"] = std

    return df