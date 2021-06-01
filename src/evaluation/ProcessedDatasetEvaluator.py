import numpy as np
from BaseEvaluator import BaseEvaluator


class ProcessedDatasetEvaluator(BaseEvaluator):
    def __init__(self, summaries, dataset, metric):
        super().__init__(metric)
        self.summaries = summaries
        self.dataset = dataset

    def evaluate(self, summary, reference_summaries):
        rc_func = self.get_rc_func()
        res = []
        for ref_summary in reference_summaries:
            rc, p_value = rc_func(summary, ref_summary)
            res.append(rc)
        #return np.asarray(res).mean()
        return np.nanmean(np.asarray(res))

