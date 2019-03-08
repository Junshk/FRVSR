import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class FHD(srdata.RAWData):
    def __init__(self, args, train=True):
        super(FHD, self).__init__(args, train)
        self.repeat = args.test_every #// (args.n_train // args.batch_size)

    def  __len__(self):
        if self.train:
            return self.args.batch_size * self.repeat
        else:
            return len(self.raw_hr)
    
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

