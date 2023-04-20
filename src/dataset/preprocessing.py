import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from typing import Optional, List, Tuple
from ..utils.utils import DictX
from ..utils.constants import JARVIS_NULL_REPLACEMENTS
from ..dataset.dataset import DataContainer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, KBinsDiscretizer, OneHotEncoder, \
    RobustScaler
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

def get_preprocessor_path(cfg: DictConfig):
    train_file_name = cfg.dataset.train.split('/')[-1].split('.')[0]
    preprocessor_applicator_file_name = f"{train_file_name}_{cfg.features.name}_{cfg.preprocessing.name}"
    preprocessor_path = Path(get_original_cwd()) / cfg.preprocessor_applicator.path / preprocessor_applicator_file_name
    return preprocessor_path


class PreprocessorApplicator(object):
    """
    Preprocessing for train or test data
    Train dataset Preprocessing: based on its own dataset
    Test dataset Preprocessing: based on train dataset
    """

    def __init__(self,
                 cfg: DictConfig,
                 X_train: pd.DataFrame,
                 y_train: Optional[pd.Series] = None,
                 num_features: List[str] = [],
                 cat_features: List[str] = [],
                 preprocessor_path: str = ""):
        self.cfg = cfg
        self.X_train = X_train.copy()
        self.y_train = y_train
        self.cat_features = cat_features
        self.num_features = num_features
        self.preprocessor_path = preprocessor_path

    def perform(self, X_pp, y_pp=None) -> DataContainer:
        """
            preprocess numerical, categorical features
        """
        # replace null
        self._replace_null_value(X_pp, self.cfg.numerical.replace_null)

        # clipping
        clipping_path = f'{self.preprocessor_path}/clipping.csv'
        if os.path.exists(clipping_path):
            self._transform_clipping(X_pp, clipping_path)

        # scaling
        scaler_path = f'{self.preprocessor_path}/scaler.pkl'
        if os.path.exists(scaler_path):
            self._transform_scaling(X_pp, scaler_path)

        X_cont = DataContainer(df=pd.DataFrame(X_pp[self.num_features], columns=self.num_features),
                               df_y=y_pp,
                               len_cat=0,
                               len_num=len(self.num_features))
        return X_cont

    def save(self):
        """
            save numerical, categorical features preprocessor
        """
        # numerical features preprocessing
        numerical_keys = self.cfg.numerical.keys() if 'numerical' in self.cfg.keys() else {}
        self._fit_num_feature(numerical_keys)

        # categorical features preprocessing
        categorical_key = self.cfg.categorical if 'categorical' in self.cfg.keys() else ''
        self._fit_cat_feature(categorical_key)

    def _transform_clipping(self, X_pp, clipping_path):
        df_clipping = pd.read_csv(clipping_path)
        df_clipping['features'] = df_clipping['features'].astype(str)
        for num_feat in self.num_features:
            X_pp[num_feat] = np.clip(X_pp[num_feat].to_numpy(),
                                     df_clipping.loc[df_clipping['features'] == num_feat]['lower_bound'].values[0],
                                     df_clipping.loc[df_clipping['features'] == num_feat]['upper_bound'].values[0])

    def _transform_scaling(self, X_pp, scaler_path):
        scaler = pickle.load(open(scaler_path, 'rb'))
        X_pp[self.num_features] = scaler.transform(X_pp[self.num_features])

    # Overall preprocessing of numerical features
    def _fit_num_feature(self, numerical_keys):
        if preprocessor_num_strategy.clipping in numerical_keys:
            self._fit_clipping(self.cfg.numerical.clipping.upper_bound,
                               self.cfg.numerical.clipping.lower_bound)
        if preprocessor_num_strategy.binning in numerical_keys:
            kbins = self._get_bins(self.cfg.numerical.binning)
            self._fit_binning(kbins)
        if preprocessor_num_strategy.scaler in numerical_keys:
            scaler = self._get_scaler(self.cfg.numerical.scaler)
            self._fit_scaler(scaler)

    # Overall preprocessing of categorical features
    def _fit_cat_feature(self, categorical_key):
        np_train_cat = self.X_train[self.cat_features].to_numpy()

        if preprocessor_cat_strategy.one_hot == categorical_key:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(self.X_train[self.cat_features])
            np_train_cat = encoder.transform(self.X_train[self.cat_features])

        elif preprocessor_cat_strategy.embedding == categorical_key:
            pass

    # scaling
    def _fit_scaler(self, scaler):
        scaler.fit(self.X_train[self.num_features])
        self.X_train[self.num_features] = scaler.transform(self.X_train[self.num_features])
        pickle.dump(scaler, open(f'{self.preprocessor_path}/scaler.pkl', 'wb'))
        print("save scaler")

    # binning
    def _fit_binning(self, kbins):
        kbins.fit(self.X_train[self.num_features].to_numpy())
        self.X_train[self.num_features] = kbins.transform(self.X_train[self.num_features].to_numpy())
        pickle.dump(kbins, open(f'{self.preprocessor_path}/binning.pkl', 'wb'))
        print("save binning")

    # clipping
    def _fit_clipping(self, upper_bound, lower_bound):
        lower_bound_list = self.X_train[self.num_features].quantile(lower_bound, numeric_only=True)
        upper_bound_list = self.X_train[self.num_features].quantile(1 - upper_bound, numeric_only=True)
        df_clipping = pd.DataFrame()
        df_clipping['features'] = self.num_features
        df_clipping['lower_bound'] = lower_bound_list.values
        df_clipping['upper_bound'] = upper_bound_list.values
        df_clipping.to_csv(f'{self.preprocessor_path}/clipping.csv')
        print("save clipping")

    # replace null value to other
    def _replace_null_value(self, X_pp, replace_value):
        for col in X_pp.columns:
            if col in JARVIS_NULL_REPLACEMENTS.keys():
                X_pp[col] = X_pp[col].replace({c: replace_value for c in JARVIS_NULL_REPLACEMENTS[col].keys()})

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
