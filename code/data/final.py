import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class FINAL(srdata.FINAL):
    def __init__(self, args, train=True):
        super(FINAL, self).__init__(args, train)    
        self.repeat = args.test_every #// (args.n_train // args.batch_size)
        self.args = args
    def  __len__(self):
        if self.train:
            return self.args.batch_size * self.repeat
        else:
            return len(self.images_hr) * self.args.n_val
    
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

