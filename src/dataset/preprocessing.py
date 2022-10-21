import numpy as np
import pandas as pd
from omegaconf import DictConfig
from typing import Optional, List, Tuple
from ..utils.utils import DictX
from ..utils.constants import JARVIS_NULL_REPLACEMENTS
from ..dataset.dataset import DataContainer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler, QuantileTransformer, KBinsDiscretizer, OneHotEncoder
from ..utils.GaussRankScaler import GaussRankScaler

preprocessor_num_strategy = DictX(
    replace_null='replace_null',
    clipping='clipping',
    scaler='scaler',
    binning='binning'
)

preprocessor_cat_strategy = DictX(
    one_hot='one_hot',
    embedding='embedding'
)
class Preprocessor():
    def __init__(self,
                 cfg: DictConfig,
                 X_train: pd.DataFrame,
                 y_train: Optional[pd.Series]=None,
                 X_test: Optional[pd.DataFrame]=None,
                 cat_features: List[str]=[]):
        self.cfg = cfg
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.cat_features = cat_features
        self.num_features = list(set(X_train.columns.tolist()) - set(cat_features))
        self.len_cat = len(self.cat_features)
        self.len_test_cat = len(self.cat_features)
        self.len_num = len(self.num_features)

    def perform(self) -> Tuple[DataContainer, Optional[DataContainer]]:
        # numerical features preprocessing
        numerical_keys = self.cfg.numerical.keys() if 'numerical' in self.cfg.keys() else {}
        np_train_num, np_test_num = self._perform_num_feature(numerical_keys)
        # categorical features preprocessing
        categorical_key = self.cfg.categorical if 'categorical' in self.cfg.keys() else ''
        np_train_cat, np_test_cat = self._perform_cat_feature(categorical_key)

        np_train = np.hstack([np_train_cat, np_train_num])
        np_test = np.hstack([np_test_cat, np_test_num])
        len_train_cat = np_train_cat.shape[1]
        len_train_num = np_train_num.shape[1]

        X_train_cont = DataContainer(df=pd.DataFrame(np_train),
                                     df_y=self.y_train,
                                     len_cat=len_train_cat,
                                     len_num=len_train_num)
        X_test_cont = None
        if self.X_test is not None:
            len_test_cat = np_test_cat.shape[1]
            len_test_num = np_test_num.shape[1]
            X_test_cont = DataContainer(df=pd.DataFrame(np_test),
                                        len_cat=len_test_cat,
                                        len_num=len_test_num)
        return X_train_cont, X_test_cont

    def _perform_num_feature(self, numerical_keys) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if preprocessor_num_strategy.replace_null in numerical_keys:
            self._replace_null_value(self.cfg.numerical.replace_null)
        if preprocessor_num_strategy.clipping in numerical_keys:
            self._fit_transform_clipping(self.cfg.numerical.clipping.upper_bound,
                                         self.cfg.numerical.clipping.lower_bound)
        if preprocessor_num_strategy.binning in numerical_keys:
            kbins = self._get_bins(self.cfg.numerical.binning)
            self._fit_transform_binning(kbins)
        if preprocessor_num_strategy.scaler in numerical_keys:
            scaler = self._get_scaler(self.cfg.numerical.scaler)
            self._fit_transform_scaler(scaler)

        np_train_num = self.X_train[self.num_features].to_numpy()
        np_test_num = None
        if self.X_test is not None:
            np_test_num = self.X_test[self.num_features].to_numpy()

        return np_train_num, np_test_num

    def _perform_cat_feature(self, categorical_key):
        np_train_cat = self.X_train[self.cat_features].to_numpy()
        np_test_cat = None
        if preprocessor_cat_strategy.one_hot == categorical_key:
            encoder = OneHotEncoder(sparse=False)
            encoder.fit(self.X_train[self.cat_features])
            np_train_cat = encoder.transform(self.X_train[self.cat_features])
            if self.X_test is not None:
                np_test_cat = encoder.transform(self.X_test[self.cat_features])
                self.len_test_cat = np_test_cat.shape[0]

        if preprocessor_cat_strategy.embedding == categorical_key:
            pass

        return np_train_cat, np_test_cat

    def _fit_transform_scaler(self, scaler):
        scaler.fit(self.X_train[self.num_features])
        self.X_train[self.num_features] = scaler.transform(self.X_train[self.num_features])
        if self.X_test is not None:
            self.X_test[self.num_features] = scaler.transform(self.X_test[self.num_features])

    def _fit_transform_binning(self, kbins):

        kbins.fit(self.X_train[self.num_features].to_numpy())
        self.X_train[self.num_features] = kbins.transform(self.X_train[self.num_features].to_numpy())
        if self.X_test is not None:
            self.X_test[self.num_features] = kbins.transform(self.X_test[self.num_features])

    def _fit_transform_clipping(self, upper_bound, lower_bound):
        lower_bound_list = self.X_train[self.num_features].quantile(lower_bound, numeric_only=True)
        upper_bound_list = self.X_train[self.num_features].quantile(1 - upper_bound, numeric_only=True)
        for num_feat in self.num_features:
            self.X_train[num_feat] = np.clip(self.X_train[num_feat].to_numpy(), lower_bound_list[num_feat], upper_bound_list[num_feat])
            if self.X_test is not None:
                self.X_test[num_feat] = np.clip(self.X_test[num_feat].to_numpy(), lower_bound_list[num_feat], upper_bound_list[num_feat])



    def _replace_null_value(self, replace_value):
        for col in self.X_train.columns:
            if col in JARVIS_NULL_REPLACEMENTS.keys():
                self.X_train[col] = self.X_train[col].replace({c: replace_value for c in JARVIS_NULL_REPLACEMENTS[col].keys()})
                if self.X_test is not None:
                    self.X_test[col] = self.X_test[col].replace({c: replace_value for c in JARVIS_NULL_REPLACEMENTS[col].keys()})

    def _get_scaler(self, scaler_cfg):
        scaler_list = {
            'quantile_transformer': QuantileTransformer(n_quantiles=scaler_cfg.n_quantiles,
                                                        output_distribution=scaler_cfg.output_distribution,
                                                        random_state=scaler_cfg.random_state),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'gauss_rank': GaussRankScaler()
        }
        return scaler_list[scaler_cfg.name]
    def _get_bins(self, binning_cfg):
        kbins = KBinsDiscretizer(n_bins=binning_cfg.n_bins,
                                 encode='ordinal',
                                 strategy=binning_cfg.strategy)
        return kbins