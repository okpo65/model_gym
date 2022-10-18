import csv
import torch
import numpy as np
import pandas as pd
from operator import lt, gt

def read_csv_file(file_path):
    res_list = []
    with open(file_path, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        res_list.append([row for row in reader][0])
    return res_list

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0,0,0,0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, percentage=False, patience=10, initial_bad=0, initial_best=np.nan, verbose=0):
        self.mode = mode
        self.patience = patience
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = initial_bad
        self.is_better = self._init_is_better(mode, min_delta, percentage)
        self.verbose = verbose
        self._stop = False

    def step(self, metric):
        if self.is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
        else:
            self.num_bad_epochs += 1

        if np.isnan(self.best) and (not np.isnan(metric)):
            self.num_bad_epochs = 0
            self.best = metric

        self._stop = self.num_bad_epochs >= self.patience
        if self.verbose and self._stop: print('Early Stopping Triggered, best score is: ', self.best)
        return self._stop

    def _init_is_better(self, mode, min_delta, percentage):
        comparator = lt if mode == 'min' else gt
        if not percentage:
            def _is_better(new, best):
                target = best - min_delta if mode == 'min' else best + min_delta
                return comparator(new, target)
        else:
            def _is_better(new, best):
                target = best * (1 - (min_delta / 100)) if mode == 'min' else best * (1 + (min_delta / 100))
                return comparator(new, target)
        return _is_better

class SwapNoiseMasker(object):
    def __init__(self, probas):
        self.probas = torch.from_numpy(np.array(probas))

    def apply(self, X):
        should_swap = torch.bernoulli(self.probas.to(X.device) * torch.ones((X.shape)).to(X.device))
        corrupted_X = torch.where(should_swap == 1, X[torch.randperm(X.shape[0])], X)
        mask = (corrupted_X != X).float()
        return corrupted_X, mask