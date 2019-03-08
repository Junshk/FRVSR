import os
import glob
import itertools

from data import common
import random
import numpy as np
import scipy.misc as misc
import pickle
import torch
import torch.utils.data as data


class FINAL(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        self.remain_idx = []
        self.idx = 0

        # self._set_filesystem(args.dir_data)
        
        
        directory = args.dir_data if train else args.dir_test
        self.images_hr, self.images_lr = self._scan(directory)
        if not  self.train: self.filename_hr, self.filename_lr = self._load_file()


    def _scan(self, dir_data):
        list_hr = []
        list_lr = [[] for _ in self.scale]

        list_bin_hr = sorted(glob.glob(os.path.join(dir_data, '*@label@*.pkl')))
        list_bin_lr = sorted(glob.glob(os.path.join(dir_data, '*@input@*.pkl')))

        
        # list_bin_hr = [hr for hr in list_bin_hr if hr.find('test')<0 and self.train or hr.find('test')>=0 and not self.train]
        #list_bin_lr = [hr.replace('label', 'input') for hr in list_bin_hr]

        list_bin_hr = [hr for hr in list_bin_hr\
            if  hr.replace('label', 'input') in list_bin_lr]
        
        return sorted(list_bin_hr), sorted([]) #raise NotImplementedError

        
    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):
        

        if len(self.remain_idx) == 0 :
            if not self.train: self.idx += 1
            self.filename_hr, self.filename_lr = self._load_file()#idx)
                
        # lr, hr, filename = self._load_file(idx)
        lr, hr, patch_idx = self._get_patch()#lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        patch_name = self.filename_hr.split('/')[-1].split('.')[0] + '_{}'.format(patch_idx[0])
        return lr_tensor, hr_tensor, patch_name #filename_hr

    def __len__(self):
        if not self.train : 
            return len(self.images_hr) * self.args.n_val
        else:
            return len(self.images_hr)*10000

    def _get_index(self, idx):
        return idx

    def _load_file(self):#, idx):
        # print('load file', self.images_hr)
        
        
        idx = random.randint(0,len(self.images_hr)-1) if self.train else self.idx
        hr = self.images_hr[idx]
        lr = hr.replace('label', 'input')
        print('id',idx)
        
        with open(hr, 'rb') as h:
            self.raw_hr = pickle.load(h)#np.fromfile(hr, 'uint8')
        with open(lr, 'rb') as l:
            self.raw_lr = pickle.load(l)#np.fromfile(lr, 'uint8')
        
        try:

            self.raw_hr = self.raw_hr.reshape(-1, 10, 144, 144, 3)
            self.raw_lr = self.raw_lr.reshape(-1, 10, 48, 48, 3)
        
        except: 
            
            self.raw_hr = np.zeros(10000, 10, 144, 144, 3)
            self.raw_lr = np.zeros(10000, 10, 48, 48, 3)

        self.remain_idx = random.sample(range(len(self.raw_hr)), len(self.raw_hr))

       	if not self.train : self.remain_idx = list(range(self.args.n_val))
        return hr, lr
        # return lr, hr, filename


    def _get_patch(self,):# lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale[0]
        multi_scale = len(self.scale) > 1
        
        
        batch_size = 1#min(self.args.batch_size, len(self.remain_idx))
        idx = self.remain_idx[:batch_size]
        
        
        self.remain_idx = self.remain_idx[batch_size:]
        lr, hr = self.raw_lr[idx], self.raw_hr[idx]

        lr = lr.astype(np.float32)[0]
        hr = hr.astype(np.float32)[0]

        # lr = lr[:,:,::-1]
        # hr = hr[:,:,::-1]

        if self.train:
            lr, hr = common.get_patch(
                lr, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr = common.augment([lr, hr])
            # lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[1:3]
            hr = hr[:, 0:ih * scale, 0:iw * scale]
        
        return lr, hr, idx

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)

        def _load_bin():
            self.images_hr = np.load(self._name_hrbin())
            self.images_lr = [
                np.load(self._name_lrbin(s)) for s in self.scale
            ]

        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr = self._scan()
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    hr = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for si, s in enumerate(self.scale):
                    for v in self.images_lr[si]:
                        lr = misc.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]

        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_hr, list_lr = self._scan()
                hr = [misc.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [misc.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                _load_bin()
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, filename

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = misc.imread(lr)
            hr = misc.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, filename

    def _get_patch(self, lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr = common.augment([lr, hr])
            lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

class RAWData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        self.remain_idx = []

        # self._set_filesystem(args.dir_data)
        
        
        self.images_hr, self.images_lr = self._scan(args.dir_data)
        if not  self.train: self._load_file()


    def _scan(self, dir_data):
        list_hr = []
        list_lr = [[] for _ in self.scale]

        list_bin_hr = glob.glob(os.path.join(dir_data,  'label*.raw'))
        list_bin_lr = glob.glob(os.path.join(dir_data, 'input*.raw'))

        
        list_bin_hr = [hr for hr in list_bin_hr if hr.find('test')<0 and self.train or hr.find('test')>=0 and not self.train]
        list_bin_lr = [lr for lr in list_bin_lr if lr.find('test')<0 and self.train or lr.find('test')>=0 and not self.train]


        return sorted(list_bin_hr), sorted(list_bin_lr) #raise NotImplementedError

        
    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):
        

        if len(self.remain_idx) == 0 and self.train:
            filename_hr, filename_lr = self._load_file()
        # lr, hr, filename = self._load_file(idx)
        lr, hr = self._get_patch()#lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, ''#filename_hr

    def __len__(self):
        return len(self.images_hr)*10000

    def _get_index(self, idx):
        return idx

    def _load_file(self):#, idx):
        # print('load file', self.images_hr)
        hr = self.images_hr[random.randint(0,len(self.images_hr)-1)]
        ID = hr[-7:-4] if self.train else hr[-8:-4]
        lr = [_ for _ in self.images_lr if _.find(ID)>=0 ][0]
        print('id',ID)

        self.raw_hr = np.fromfile(hr, 'uint8')
        self.raw_lr = np.fromfile(lr, 'uint8')

        self.raw_hr = self.raw_hr.reshape(-1, 288, 288, 3)
        self.raw_lr = self.raw_lr.reshape(-1, 96, 96, 3)
        

        self.remain_idx = random.sample(range(len(self.raw_hr)), len(self.raw_hr))
        if not self.train : self.remain_idx = list(range(len(self.raw_hr)))
        '''
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = misc.imread(lr)
            hr = misc.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]
        '''
        return hr, lr
        # return lr, hr, filename


    def _get_patch(self,):# lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale[0]
        multi_scale = len(self.scale) > 1

        
        batch_size = 1#min(self.args.batch_size, len(self.remain_idx))
        # scale = self.scale[self.idx_scale]
        # multi_scale = len(self.scale) > 1
        idx = self.remain_idx[:batch_size]
        
        self.remain_idx = self.remain_idx[batch_size:]
        lr, hr = self.raw_lr[idx], self.raw_hr[idx]

        lr = lr.astype(np.float32)[0]
        hr = hr.astype(np.float32)[0]

        lr = lr[:,:,::-1]
        hr = hr[:,:,::-1]

        # lr = lr.transpose(2,0,1)
        # hr = hr.transpose(2,0,1)

        if self.train:
            lr, hr = common.get_patch(
                lr, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr = common.augment([lr, hr])
            # lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
