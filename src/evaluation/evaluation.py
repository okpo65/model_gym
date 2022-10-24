from typing import Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import roc_curve as r_curve
from sklearn.metrics import r2_score
from pytorch_tabnet.metrics import Metric
import matplotlib.pyplot as plt
import seaborn as sns


class AmexMetric(Metric):
    def __init__(self):
        self._name = "amex-metric"
        self._maximize = True

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        score = amex_metric(y_true, y_score[:, 1])
        return score


class CatBoostEvalMetricAmex:
    def get_final_error(self, error: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return error

    def is_max_optimal(self) -> bool:
        return True

    def evaluate(
        self, approxes: np.ndarray, target: np.ndarray, weight: np.ndarray
    ) -> Tuple[float, int]:
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        preds = np.array(approxes[0])
        target = np.array(target)
        return amex_metric(np.array(target), np.array(preds)), 0


def amex_metric(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> float:
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])
    gini = [0, 0]

    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


def lgb_amex_metric(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[str, float, bool]:
    """The competition metric with lightgbm's calling convention"""
    y_true = y_true.get_label()
    return "amex", amex_metric(y_true, y_pred), True


def xgb_amex_metric(y_pred: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """The competition metric with xgboost's calling convention"""
    y_true = dtrain.get_label()
    return "amex", amex_metric(y_true, y_pred)


class SegMetric:
    def __init__(self, task='css'):
        self.task = task
        self.num = 100

    def ks(self, default_prob, y_true, ax, name):
        assert default_prob.shape == y_true.shape
        restable = pd.DataFrame({'default': y_true})
        restable['nondefault'] = 1 - restable['default']
        restable['prob1'] = default_prob
        restable['prob0'] = 1 - default_prob
        restable['bucket'] = pd.qcut(restable['prob0'], self.num, duplicates='drop')
        grouped = restable.groupby('bucket', as_index=False)
        kstable = pd.DataFrame()
        kstable['min_prob'] = grouped.min()['prob0']
        kstable['max_prob'] = grouped.max()['prob0']
        kstable['defaults'] = grouped.sum()['default']
        kstable['nondefaults'] = grouped.sum()['nondefault']
        kstable = kstable.sort_values(by='min_prob', ascending=True).reset_index(drop=True)
        kstable['default_rate'] = (kstable.defaults / restable['default'].sum()).apply('{0:.2%}'.format)
        kstable['nondefault_rate'] = (kstable.nondefaults / restable['nondefault'].sum()).apply('{0:.2%}'.format)
        kstable['cum_defaultrate'] = (kstable.defaults / restable['default'].sum()).cumsum()
        kstable['cum_nondefaultrate'] = (kstable.nondefaults / restable['nondefault'].sum()).cumsum()
        kstable['KS'] = np.round(kstable['cum_defaultrate'] - kstable['cum_nondefaultrate'], 3) * 100

        def draw_max_ks(x):
            ax.plot([x.min_prob, x.min_prob], [x.cum_nondefaultrate, x.cum_defaultrate], color='#FA1600')
            print(f'[{name}] KS is {x.KS}% at score {x.min_prob}')

        max_row = kstable.loc[kstable['KS'] == max(kstable['KS']), :]
        max_row = max_row.copy()
        max_row['min_prob'] = max_row['min_prob'].map(lambda x: round(x, 2))
        max_row = max_row.drop_duplicates(['min_prob'])
        max_row.head(1).apply(draw_max_ks, axis=1)
        sns.lineplot(x=kstable['min_prob'], y=kstable.cum_defaultrate, label='Default', color='#000000', ax=ax)
        sns.lineplot(x=kstable['min_prob'], y=kstable.cum_nondefaultrate, label='Non-Default', color='#265BFA', ax=ax)
        ax.set_ylabel('Cumulative rate')
        ax.set_xlabel('Score')
        ax.set(ylim=(0, 1))
        ax.set_title('[' + name + ']: KS Score')
        kstable['cum_defaultrate'] = kstable['cum_defaultrate'].apply('{0:.2%}'.format)
        kstable['cum_nondefaultrate'] = kstable['cum_nondefaultrate'].apply('{0:.2%}'.format)
        return kstable

    def get_ks_table(self, default_prob, y_true):
        restable = self.get_res_table(default_prob, y_true)
        grouped = restable.groupby('bucket', as_index=False)
        kstable = pd.DataFrame()
        kstable['min_prob'] = grouped.min()['prob0']
        kstable['max_prob'] = grouped.max()['prob0']
        kstable['defaults'] = grouped.sum()['default']
        kstable['nondefaults'] = grouped.sum()['nondefault']
        kstable = kstable.sort_values(by='min_prob', ascending=True).reset_index(drop=True)
        kstable['default_rate'] = (kstable.defaults / restable['default'].sum()).apply('{0:.2%}'.format)
        kstable['nondefault_rate'] = (kstable.nondefaults / restable['nondefault'].sum()).apply('{0:.2%}'.format)
        kstable['cum_defaultrate'] = (kstable.defaults / restable['default'].sum()).cumsum()
        kstable['cum_nondefaultrate'] = (kstable.nondefaults / restable['nondefault'].sum()).cumsum()
        kstable['KS'] = np.round(kstable['cum_defaultrate'] - kstable['cum_nondefaultrate'], 3) * 100
        #         kstable['cum_defaultrate'] = kstable['cum_defaultrate'].apply('{0:.2%}'.format)
        #         kstable['cum_nondefaultrate'] = kstable['cum_nondefaultrate'].apply('{0:.2%}'.format)
        return kstable

    def get_res_table(self, default_prob, y_true):
        assert default_prob.shape == y_true.shape
        restable = pd.DataFrame({'default': y_true})
        restable['nondefault'] = 1 - restable['default']
        restable['prob1'] = default_prob
        restable['prob0'] = 1 - default_prob
        restable['bucket'] = pd.qcut(restable['prob0'], self.num, duplicates='drop')

        return restable

    def get_max_ks(self, default_prob, y_true):
        kstable = self.get_ks_table(default_prob, y_true)
        max_row = kstable.loc[kstable['KS'] == max(kstable['KS']), :]
        max_row = max_row.copy()
        max_row['min_prob'] = max_row['min_prob'].map(lambda x: round(x, 2))
        return max_row.KS.head(1).values[0]

    def roc_curve(self, y_pred, y_true, ax, name, pos_label=1):
        assert y_true.shape == y_pred.shape
        fpr, tpr, thresholds = r_curve(y_true, y_pred, pos_label=pos_label)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=self.task)
        display.plot(ax=ax)
        ax.set_title('[' + name + ']: ROC Curve')
        print(f'[{name}] ROC_AUC : {roc_auc}')

    def get_roc_curve(self, default_prob, y_true, pos_label=1):
        #         y_true = y_true['CSS_TARGET']

        fpr, tpr, thresholds = r_curve(y_true, default_prob, pos_label=pos_label)
        roc_auc = metrics.auc(fpr, tpr)
        return roc_auc

    def draw_ks(self, default_prob, y_true, ax, name):
        def draw_max_ks(x):
            ax.plot([x.min_prob, x.min_prob], [x.cum_nondefaultrate, x.cum_defaultrate], color='#FA1600')

        #             print(f'[{name}] KS is {x.KS}% at score {x.min_prob}')

        kstable = self.get_ks_table(default_prob, y_true)
        max_row = kstable.loc[kstable['KS'] == max(kstable['KS']), :]
        max_row = max_row.copy()
        max_row['min_prob'] = max_row['min_prob'].map(lambda x: round(x, 2))

        max_row = max_row.drop_duplicates(['min_prob'])
        max_row.head(1).apply(draw_max_ks, axis=1)
        sns.lineplot(x=kstable['min_prob'], y=kstable.cum_defaultrate, label='Default', color='#000000', ax=ax)
        sns.lineplot(x=kstable['min_prob'], y=kstable.cum_nondefaultrate, label='Non-Default', color='#265BFA', ax=ax)
        ax.set_ylabel('Cumulative rate')
        ax.set_xlabel('Score')
        ax.set(ylim=(0, 1))
        ax.set_title('[' + name + ']: KS Score')


class CSSTask:
    def eval_seg(self, prediction, score):
        data_path = '/data/jarvis/css_v4'
        y_test = pd.read_csv(os.path.join(data_path, 'test', 'y_test.csv'))
        y_test = y_test['CSS_TARGET']
        y_flag = pd.read_csv(os.path.join(data_path, 'test', 'flag.csv'))
        # y_flag = self._load_data(dir_name='test', file_name='flag.csv')
        if score.shape != y_test.shape:
            if score.shape[1] == 2:
                score = score[:, 1]
            elif score.shape[1] == 1:
                score = score.reshape(-1)
        assert score.shape == y_test.shape
        df = pd.DataFrame({'prediction': prediction,
                           'score': score,
                           'y_test': y_test,
                           'kakao': y_flag['kakao'],
                           'grd_1_3': y_flag['grd_1_3'],
                           'grd_4_7': y_flag['grd_4_7'],
                           'grd_8_10': y_flag['grd_8_10'],
                           'LS0000175': y_flag['LS0000175']})
        df_kakao = df.loc[df['kakao'] == 1]
        df_1_3 = df.loc[df['grd_1_3'] == 1]
        df_4_7 = df.loc[df['grd_4_7'] == 1]
        df_8_10 = df.loc[df['grd_8_10'] == 1]
        df_ls0000175 = df.loc[df['LS0000175'] == 1]
        fig, axes = plt.subplots(ncols=6, figsize=(20, 6))
        fig2, axes2 = plt.subplots(ncols=6, figsize=(20, 6))
        metric = SegMetric(task='css')
        print('-----------------------------------------------')
        metric.ks(df['score'], y_test, ax=axes[0], name='all')
        metric.ks(df_kakao['score'], df_kakao['y_test'], ax=axes[1], name='kakao')
        metric.ks(df_1_3['score'], df_1_3['y_test'], ax=axes[2], name='grd_1~3')
        metric.ks(df_4_7['score'], df_4_7['y_test'], ax=axes[3], name='grd_4~7')
        metric.ks(df_8_10['score'], df_8_10['y_test'], ax=axes[4], name='grd_8~10')
        metric.ks(df_ls0000175['score'], df_ls0000175['y_test'], ax=axes[5], name='2DN')
        print('-----------------------------------------------')
        metric.roc_curve(df['score'], y_test, ax=axes2[0], name='all')
        metric.roc_curve(df_kakao['score'], df_kakao['y_test'], ax=axes2[1], name='kakao')
        metric.roc_curve(df_1_3['score'], df_1_3['y_test'], ax=axes2[2], name='grd_1~3')
        metric.roc_curve(df_4_7['score'], df_4_7['y_test'], ax=axes2[3], name='grd_4~7')
        metric.roc_curve(df_8_10['score'], df_8_10['y_test'], ax=axes2[4], name='grd_8~10')
        metric.roc_curve(df_ls0000175['score'], df_ls0000175['y_test'], ax=axes2[5], name='2DN')
        print('-----------------------------------------------')
        fig.canvas.draw()
        fig2.canvas.draw()

    def eval_seg_v2(self, score, y_test, name='all', target_name='CSS_TARGET'):
        y_test = y_test[target_name]
        if score.shape != y_test.shape:
            if score.shape[1] == 2:
                score = score[:, 1]
            elif score.shape[1] == 1:
                score = score.reshape(-1)
        assert score.shape == y_test.shape
        df = pd.DataFrame({'score': score})
        metric = SegMetric(task='css')
        fig, axes = plt.subplots(ncols=1, figsize=(6, 6))
        metric.ks(df['score'], y_test, ax=axes, name=name)

    def eval_seg_ks(self, score, y_test, target_name='CSS_TARGET'):
        y_test = y_test[target_name]
        if score.shape != y_test.shape:
            if score.shape[1] == 2:
                score = score[:, 1]
            elif score.shape[1] == 1:
                score = score.reshape(-1)
        assert score.shape == y_test.shape
        df = pd.DataFrame({'score': score})
        metric = SegMetric(task='css')
        return metric.get_max_ks(df['score'], y_test)


    def draw_seg_ks(self, score, y_test, target_name='CSS_TARGET'):
        y_test = y_test[target_name]
        if score.shape != y_test.shape:
            if score.shape[1] == 2:
                score = score[:, 1]
            elif score.shape[1] == 1:
                score = score.reshape(-1)
        assert score.shape == y_test.shape
        df = pd.DataFrame({'score': score})
        metric = SegMetric(task='css')
        fig, axes = plt.subplots(ncols=1, figsize=(6, 6))
        metric.draw_ks(df['score'], y_test, axes, "all")

def css_metric(y_pred: Union[pd.Series, np.ndarray],
               y_true: Union[pd.Series, np.ndarray]) -> float:
    return CSSTask().eval_seg_ks(y_pred,
                                 pd.DataFrame({'target':y_true}).reset_index(drop=True),
                                 'target')
