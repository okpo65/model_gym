import pandas as pd
from omegaconf import DictConfig
from typing import Tuple, Optional
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


class DataContainer():
    def __init__(self,
                 df: pd.DataFrame,
                 df_y: Optional[pd.Series]=None,
                 len_cat: int=0,
                 len_num: int=0):
        """
        :param df: preprocessed data frame
        :param len_cat: count of categorical features
        :param len_num: count of numerical features
        """
        self.df = df
        self.df_y = df_y
        self.len_cat = len_cat
        self.len_num = len_num

    def _split_dataset(self, _df, cutoff_ratio):
        cutoff = int(len(_df) * cutoff_ratio)
        train_data = _df.iloc[:cutoff].to_numpy()
        valid_data = _df.iloc[cutoff:].to_numpy()

        return train_data, valid_data

    def get_dae_dataset(self, batch_size, num_workers):
        train_x, valid_x = self._split_dataset(self.df, 0.9)
        train_x = DataLoader(dataset=DAEDataset(train_x),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True)
        valid_x = DataLoader(dataset=DAEDataset(valid_x),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False)
        return train_x, valid_x

    def get_dl_dataloader_for_training(self, batch_size, num_workers):
        train_x, valid_x = self._split_dataset(self.df, 0.9)
        train_y, valid_y = self._split_dataset(self.df_y, 0.9)
        train_x = DataLoader(dataset=TrainDataset(train_x, train_y),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True)
        valid_x = DataLoader(dataset=TrainDataset(valid_x, valid_y),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False)
        return train_x, valid_x

    def get_dl_dataloader_for_testing(self, batch_size, num_workers):
        test_x = DataLoader(dataset=TestDataset(self.df.to_numpy()),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False)

        return test_x

    def get_dataframe(self):
        return self.df, self.df_y

    # @property
    # def len_cat(self): return self.len_cat
    #
    # @property
    # def len_num(self): return self.len_num


def load_train_data(config: DictConfig) -> Tuple[pd.DataFrame, pd.Series]:
    feat_list = read_csv_file(config.dataset.feature_list_path)[0]
    cat_feat_list = read_csv_file(config.dataset.cat_feature_path)[0]

    X_train = pd.read_parquet(config.dataset.train)
    # random shuffle
    X_train = X_train.sample(frac=1.0).reset_index(drop=True)
    y_train = X_train[config.dataset.target_name]

    return X_train[feat_list], y_train

def load_test_data(config: DictConfig) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    feat_list = read_csv_file(config.dataset.feature_list_path)[0]
    cat_feat_list = read_csv_file(config.dataset.cat_feature_path)[0]


    X_test = pd.read_parquet(config.dataset.test)

    target_name = config.dataset.target_name
    if target_name in X_test.columns:
        y_test = X_test[target_name]
    else:
        y_test = None
    return X_test[feat_list], y_test




