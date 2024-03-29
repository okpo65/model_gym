import pandas as pd
from omegaconf import DictConfig
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

        Overall Data Repository with feature information
        """
        self.df = df
        self.df_y = df_y
        self.len_cat = len_cat
        self.len_num = len_num

    def _split_dataframe(self, _df, cutoff_ratio) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cutoff = int(len(_df) * cutoff_ratio)
        train_data = _df.iloc[:cutoff]
        valid_data = _df.iloc[cutoff:].reset_index(drop=True)

        return train_data, valid_data

    def get_dae_dataloader(self, batch_size, num_workers) -> Tuple[DataLoader, DataLoader]:
        train_x, valid_x = self._split_dataframe(self.df, 0.9)
        train_x = DataLoader(dataset=DAEDataset(train_x.to_numpy()),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True)
        valid_x = DataLoader(dataset=DAEDataset(valid_x.to_numpy()),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False)
        return train_x, valid_x

    def get_train_dataloader(self, batch_size, num_workers) -> DataLoader:
        x_dl = DataLoader(dataset=TrainDataset(self.df.to_numpy(), self.df_y),
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)
        return x_dl

    def get_valid_dataloader(self, batch_size, num_workers) -> DataLoader:
        x_dl = DataLoader(dataset=TrainDataset(self.df.to_numpy(), self.df_y),
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=False)
        return x_dl

    def get_test_dataloader(self, batch_size, num_workers) -> DataLoader:
        test_x = DataLoader(dataset=TestDataset(self.df.to_numpy()),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False)

        return test_x

    def get_dataframe(self):
        return self.df, self.df_y

    def get_splited_data_series(self, split_ratio=0.9) -> Tuple[pd.DataFrame,
                                                                pd.DataFrame,
                                                                pd.Series,
                                                                pd.Series]:
        train_x, valid_x = self._split_dataframe(self.df, split_ratio)
        train_y, valid_y = self._split_dataframe(self.df_y, split_ratio)
        return train_x, valid_x, train_y.to_numpy(), valid_y

    def get_splited_dataloader(self, batch_size, num_workers, split_ratio=0.9) -> Tuple[DataLoader,
                                                                                        DataLoader]:
        train_x, valid_x, train_y, valid_y = self.get_splited_data_series(split_ratio)

        train_x = DataLoader(dataset=TrainDataset(train_x.to_numpy(), train_y),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True)
        valid_x = DataLoader(dataset=TrainDataset(valid_x.to_numpy(), valid_y),
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False)
        return train_x, valid_x

def load_train_data(config: DictConfig) -> Tuple[pd.DataFrame,
                                                 pd.Series]:
    """
    :param config: dataset config
    :return: shuffled train dataset
    """
    feat_list = [*config.features.total_features]
    if config.dataset.train.split('.')[-1] == 'parquet':
        X_train = pd.read_parquet(config.dataset.train)
    else:
        X_train = pd.read_csv(config.dataset.train)
    # random shuffle
    X_train = X_train.fillna(0)
    y_train = X_train[config.dataset.target_name]

    return X_train[feat_list], y_train

def load_test_data(config: DictConfig,
                   test_path: Optional[str]=None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    :param config: dataset config
    :return: test dataset
    """
    feat_list = [*config.features.total_features]
    if test_path is not None:
        test_data_path = test_path
    else:
        test_data_path = config.dataset.test

    if test_data_path.split('.')[-1] == 'parquet':
        X_test = pd.read_parquet(test_data_path)
    else:
        X_test = pd.read_csv(test_data_path)
    X_test = X_test.fillna(0)
    target_name = config.dataset.target_name
    if target_name in X_test.columns:
        y_test = X_test[target_name]
    else:
        y_test = None
    return X_test[feat_list], y_test




