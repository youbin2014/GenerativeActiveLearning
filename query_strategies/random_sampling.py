import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset, net, args_input, args_task,diffuser):
        super(RandomSampling, self).__init__(dataset, net, args_input, args_task,diffuser)

    def query(self, n):
        return np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n, replace=False)
