import pandas as pd
from omegaconf import DictConfig
from typing import Tuple
from ..utils.utils import read_csv_file
from torch.utils.data import Dataset, DataLoader
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

class TrainDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x.astype('float32'), y.astype('float32')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class TestDataset(Dataset):
    def __init__(self, x):
        self.x = x.astype('float32')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]

class DAEDataset(Dataset):
    def __init__(self, x):
        self.x = x.astype('float32')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]


def get_dae_dataset(X: np.ndarray,
                    batch_size: int,
                    num_workers: int) -> Dataset:
    dataloader = DataLoader(dataset=DAEDataset(X),
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    return dataloader

def get_dataset_with_cat_handler(X: pd.DataFrame,
                                 cat_list: List[str]) -> (np.ndarray, int, int):
    num_list = list(set(X.columns) - set(cat_list))
    encoder = OneHotEncoder(sparse=False)
    X_cat = encoder.fit_transform(X[cat_list])
    encoder = StandardScaler()
    X_num = encoder.fit_transform(X[num_list])
    X = np.hstack([X_cat, X_num])
    return X, X_cat.shape[1], X_num.shape[1]

def load_train_data(config: DictConfig) -> Tuple[pd.DataFrame, pd.Series]:
    feat_list = read_csv_file(config.dataset.feature_list_path)[0]
    cat_feat_list = read_csv_file(config.dataset.cat_feature_path)[0]

    X_train = pd.read_parquet(config.dataset.train)
    y_train = X_train[config.dataset.target_name]

    return X_train[feat_list], y_train

def load_test_data(config: DictConfig):
    feat_list = read_csv_file(config.dataset.feature_list_path)[0]
    cat_feat_list = read_csv_file(config.dataset.cat_feature_path)[0]

    target_name = config.dataset.target_name
    X_test = pd.read_parquet(config.dataset.test)
    if target_name in X_test.columns:
        y_test = X_test[target_name]
    else:
        y_test = None
    return X_test[feat_list], y_test




