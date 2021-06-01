import abc

from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata


class BaseEvaluator(abc.ABC):
    def __init__(self, metric):
        self.metric = metric

    @abc.abstractmethod
    def evaluate(self):
        pass

    def get_rc_func(self):

        if self.metric == "kendalltau":
            print('kendalltau')
            f = lambda x, y: kendalltau(rankdata(-x), rankdata(-y), nan_policy='omit')
        elif self.metric == "spearmanr":
            print('spearmanr')
            f = lambda x, y: spearmanr(x, y)
        else:
            raise RuntimeError
        return f
