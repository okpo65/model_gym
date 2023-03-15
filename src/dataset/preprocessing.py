import pickle

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from typing import Optional, List, Tuple
from ..utils.utils import DictX
from ..utils.constants import JARVIS_NULL_REPLACEMENTS
from ..dataset.dataset import DataContainer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, KBinsDiscretizer, OneHotEncoder, RobustScaler
from ..utils.GaussRankScaler import GaussRankScaler


# available numerical preprocessing strategy
preprocessor_num_strategy = DictX(
    replace_null='replace_null',
    clipping='clipping',
    scaler='scaler',
    binning='binning'
)

# available categorical preprocessing strategy
preprocessor_cat_strategy = DictX(
    one_hot='one_hot',
    embedding='embedding'
)

class Preprocessor(object):
    """
    Preprocessing for train or test data
    Train dataset Preprocessing: based on its own dataset
    Test dataset Preprocessing: based on train dataset
    """
    def __init__(self,
                 cfg: DictConfig,
                 X_train: pd.DataFrame,
                 y_train: Optional[pd.Series]=None,
                 X_test: Optional[pd.DataFrame]=None,
                 num_features: List[str]=[],
                 cat_features: List[str]=[]):
        self.cfg = cfg
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.cat_features = cat_features
        self.num_features = num_features

        self.X_train = self.X_train.fillna(0)
        if self.X_test is not None:
            self.X_test = self.X_test.fillna(0)

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

        train_columns = [f"cat_{idx}" for idx in range(0, len_train_cat)] + [col for col in self.num_features]
        X_train_cont = DataContainer(df=pd.DataFrame(np_train, columns=train_columns),
                                     df_y=self.y_train,
                                     len_cat=len_train_cat,
                                     len_num=len_train_num)
        X_test_cont = None
        if self.X_test is not None:
            len_test_cat = np_test_cat.shape[1]
            len_test_num = np_test_num.shape[1]
            test_columns = [f"cat_{idx}" for idx in range(0, len_test_cat)] + [col for col in self.num_features]
            X_test_cont = DataContainer(df=pd.DataFrame(np_test, columns=test_columns),
                                        len_cat=len_test_cat,
                                        len_num=len_test_num)
        return X_train_cont, X_test_cont

    # Overall preprocessing of numerical features
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

    # Overall preprocessing of categorical features
    def _perform_cat_feature(self, categorical_key):
        np_train_cat = self.X_train[self.cat_features].to_numpy()
        if self.X_test is not None:
            np_test_cat = self.X_test[self.cat_features].to_numpy()
        else:
            np_test_cat = None

        if preprocessor_cat_strategy.one_hot == categorical_key:
            encoder = OneHotEncoder(sparse=False, handle_unknown = 'ignore')
            encoder.fit(self.X_train[self.cat_features])
            np_train_cat = encoder.transform(self.X_train[self.cat_features])
            if self.X_test is not None:
                np_test_cat = encoder.transform(self.X_test[self.cat_features])

        elif preprocessor_cat_strategy.embedding == categorical_key:
            pass

        return np_train_cat, np_test_cat

    # scaling
    def _fit_transform_scaler(self, scaler):
        scaler.fit(self.X_train[self.num_features])
        self.X_train[self.num_features] = scaler.transform(self.X_train[self.num_features])
        if self.X_test is not None:
            self.X_test[self.num_features] = scaler.transform(self.X_test[self.num_features])

    # binning
    def _fit_transform_binning(self, kbins):
        kbins.fit(self.X_train[self.num_features].to_numpy())
        self.X_train[self.num_features] = kbins.transform(self.X_train[self.num_features].to_numpy())
        if self.X_test is not None:
            self.X_test[self.num_features] = kbins.transform(self.X_test[self.num_features])

    # clipping
    def _fit_transform_clipping(self, upper_bound, lower_bound):
        lower_bound_list = self.X_train[self.num_features].quantile(lower_bound, numeric_only=True)
        upper_bound_list = self.X_train[self.num_features].quantile(1 - upper_bound, numeric_only=True)
        for num_feat in self.num_features:
            self.X_train[num_feat] = np.clip(self.X_train[num_feat].to_numpy(), lower_bound_list[num_feat], upper_bound_list[num_feat])
            if self.X_test is not None:
                self.X_test[num_feat] = np.clip(self.X_test[num_feat].to_numpy(), lower_bound_list[num_feat], upper_bound_list[num_feat])

    # replace null value to other
    def _replace_null_value(self, replace_value):
        for col in self.X_train.columns:
            if col in JARVIS_NULL_REPLACEMENTS.keys():
                self.X_train[col] = self.X_train[col].replace({c: replace_value for c in JARVIS_NULL_REPLACEMENTS[col].keys()})
                if self.X_test is not None:
                    self.X_test[col] = self.X_test[col].replace({c: replace_value for c in JARVIS_NULL_REPLACEMENTS[col].keys()})

    # get scaler
    def _get_scaler(self, scaler_cfg):
        scaler_list = {
            'quantile_transformer': QuantileTransformer(n_quantiles=scaler_cfg.n_quantiles,
                                                        output_distribution=scaler_cfg.output_distribution,
                                                        random_state=scaler_cfg.random_state),
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'gauss_rank': GaussRankScaler()
        }
        return scaler_list[scaler_cfg.name]

    # get binning object
    def _get_bins(self, binning_cfg):
        kbins = KBinsDiscretizer(n_bins=binning_cfg.n_bins,
                                 encode='ordinal',
                                 strategy=binning_cfg.strategy)
        return kbins

    # def _fit_sacler(self, scaler):
    #     scaler.fit(self.X_train[self.num_features])
    #
    # def _fit_binnning(self, kbins):
    #     kbins.fit(self.X_train[self.num_features].to_numpy())
    #
    # def _fit_clipping(self, upper_bound, lower_bound):
    #     lower_bound_list = self.X_train[self.num_features].quantile(lower_bound, numeric_only=True)
    #     upper_bound_list = self.X_train[self.num_features].quantile(1 - upper_bound, numeric_only=True)
    #
    # def _replace_null(self):


class Preprocessor2(object):
    """
    Preprocessing for train or test data
    Train dataset Preprocessing: based on its own dataset
    Test dataset Preprocessing: based on train dataset
    """
    def __init__(self,
                 X_test: pd.DataFrame,
                 num_features: List[str]=[]):
        self.X_test = X_test
        self.num_features = num_features

    def perform(self) -> DataContainer:
        # numerical features preprocessing
        # replace null
        for col in self.num_features:
            if col in JARVIS_NULL_REPLACEMENTS.keys():
                self.X_test[col] = self.X_test[col].replace({c: 0 for c in JARVIS_NULL_REPLACEMENTS[col].keys()})
        # clipping
        df_tmp = pd.read_csv('preprocessor_clipping.csv')
        for num_feat in self.num_features:
            self.X_test[num_feat] = np.clip(self.X_test[num_feat].to_numpy(),
                                            df_tmp.loc[df_tmp['features'] == num_feat]['lower_bound'].values[0],
                                            df_tmp.loc[df_tmp['features'] == num_feat]['upper_bound'].values[0])

        # scaling
        scaler = pickle.load(open('preprocessor_scaler.pkl', 'rb'))
        self.X_test[self.num_features] = scaler.transform(self.X_test[self.num_features])

        np_test = self.X_test.to_numpy()

        train_columns = [col for col in self.num_features]
        X_test_cont = DataContainer(df=pd.DataFrame(np_test, columns=train_columns),
                                     len_cat=0,
                                     len_num=len(self.num_features))
        return X_test_cont

    # Overall preprocessing of numerical features
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

    # Overall preprocessing of categorical features
    def _perform_cat_feature(self, categorical_key):
        np_train_cat = self.X_train[self.cat_features].to_numpy()
        if self.X_test is not None:
            np_test_cat = self.X_test[self.cat_features].to_numpy()
        else:
            np_test_cat = None

        if preprocessor_cat_strategy.one_hot == categorical_key:
            encoder = OneHotEncoder(sparse=False, handle_unknown = 'ignore')
            encoder.fit(self.X_train[self.cat_features])
            np_train_cat = encoder.transform(self.X_train[self.cat_features])
            if self.X_test is not None:
                np_test_cat = encoder.transform(self.X_test[self.cat_features])

        elif preprocessor_cat_strategy.embedding == categorical_key:
            pass

        return np_train_cat, np_test_cat

    # scaling
    def _fit_transform_scaler(self, scaler):
        scaler.fit(self.X_train[self.num_features])
        self.X_train[self.num_features] = scaler.transform(self.X_train[self.num_features])
        if self.X_test is not None:
            self.X_test[self.num_features] = scaler.transform(self.X_test[self.num_features])
        pickle.dump(scaler, open('preprocessor_scaler.pkl', 'wb'))
        print("save scaler")
    # binning
    def _fit_transform_binning(self, kbins):
        kbins.fit(self.X_train[self.num_features].to_numpy())
        self.X_train[self.num_features] = kbins.transform(self.X_train[self.num_features].to_numpy())
        if self.X_test is not None:
            self.X_test[self.num_features] = kbins.transform(self.X_test[self.num_features])

    # clipping
    def _fit_transform_clipping(self, upper_bound, lower_bound):
        lower_bound_list = self.X_train[self.num_features].quantile(lower_bound, numeric_only=True)
        upper_bound_list = self.X_train[self.num_features].quantile(1 - upper_bound, numeric_only=True)

        df_tmp = pd.DataFrame()
        df_tmp['features'] = self.num_features
        df_tmp['lower_bound'] = lower_bound_list
        df_tmp['upper_bound'] = upper_bound_list
        print("save clipping")
        df_tmp.to_csv('preprocessor_clipping.csv')
        for num_feat in self.num_features:
            self.X_train[num_feat] = np.clip(self.X_train[num_feat].to_numpy(), lower_bound_list[num_feat], upper_bound_list[num_feat])
            if self.X_test is not None:
                self.X_test[num_feat] = np.clip(self.X_test[num_feat].to_numpy(), lower_bound_list[num_feat], upper_bound_list[num_feat])

    # replace null value to other
    def _replace_null_value(self, replace_value):
        for col in self.X_train.columns:
            if col in JARVIS_NULL_REPLACEMENTS.keys():
                self.X_train[col] = self.X_train[col].replace({c: replace_value for c in JARVIS_NULL_REPLACEMENTS[col].keys()})
                if self.X_test is not None:
                    self.X_test[col] = self.X_test[col].replace({c: replace_value for c in JARVIS_NULL_REPLACEMENTS[col].keys()})

    # get scaler
    def _get_scaler(self, scaler_cfg):
        scaler_list = {
            'quantile_transformer': QuantileTransformer(n_quantiles=scaler_cfg.n_quantiles,
                                                        output_distribution=scaler_cfg.output_distribution,
                                                        random_state=scaler_cfg.random_state),
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'gauss_rank': GaussRankScaler()
        }
        return scaler_list[scaler_cfg.name]

    # get binning object
    def _get_bins(self, binning_cfg):
        kbins = KBinsDiscretizer(n_bins=binning_cfg.n_bins,
                                 encode='ordinal',
                                 strategy=binning_cfg.strategy)
        return kbins

    # def _fit_sacler(self, scaler):
    #     scaler.fit(self.X_train[self.num_features])
    #
    # def _fit_binnning(self, kbins):
    #     kbins.fit(self.X_train[self.num_features].to_numpy())
    #
    # def _fit_clipping(self, upper_bound, lower_bound):
    #     lower_bound_list = self.X_train[self.num_features].quantile(lower_bound, numeric_only=True)
    #     upper_bound_list = self.X_train[self.num_features].quantile(1 - upper_bound, numeric_only=True)
    #
    # def _replace_null(self):