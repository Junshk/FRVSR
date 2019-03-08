import os
import math

import utility
from data import common

import torch
import cv2
import numpy as np
from tqdm import tqdm

class VideoTester():
    def __init__(self, args, my_model, ckp):
        self.args = args
        self.scale = args.scale

        #self.ckp = ckp
        self.model = my_model

        self.filename, _ = os.path.splitext(os.path.basename(args.dir_demo))

    def test(self):
        torch.set_grad_enabled(False)

        # self.ckp.write_log('\nEvaluation on video:')
        self.model.eval()

        timer_test = utility.timer()
        for idx_scale, scale in enumerate(self.scale):
            vidcap = cv2.VideoCapture(self.args.dir_demo)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

            W = int(scale * vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(scale * vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            vidwri = cv2.VideoWriter(
                '{}/{}_x{}.avi'.format(self.args.save_dir, self.filename, scale),
                cv2.VideoWriter_fourcc(*'XVID'),
                vidcap.get(cv2.CAP_PROP_FPS),
                (
                    W,#int(scale * vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    H,#int(scale * vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
            )

            tqdm_test = tqdm(range(total_frames), ncols=80)
            lr = torch.zeros([1, 3, H//scale, W//scale])
            estimate = torch.zeros(1, 3, H, W)

            for _ in tqdm_test:
                lr_ = lr
                success, lr = vidcap.read()
                if not success: break
                
                lr = (common.set_channel([lr], n_channel=self.args.n_colors))
                lr = (common.np2Tensor(np.array([lr]), rgb_range=self.args.rgb_range))
                lr, lr_, estimate = self.prepare([lr[0], lr_, estimate])

                with torch.no_grad():
                    sr, lre = self.model(lr, lr_, estimate)#forward_chop(lr[0], self.model, scale=scale)#self.model(lr[0], idx_scale)
                    estimate = sr
                sr = utility.quantize(sr, self.args.rgb_range).squeeze(0)

                normalized = sr * 255 / self.args.rgb_range
                ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                vidwri.write(ndarr)

            vidcap.release()
            vidwri.release()
        '''
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )'''
        torch.set_grad_enabled(True)

    def prepare(self, args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

def forward_chop(x, model, shave=10, min_size=160000, scale=1):
        # scale = #self.scale[self.idx_scale]
        n_GPUs = 1#min(2, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = model(lr_batch, scale)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                forward_chop(patch, model, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output
