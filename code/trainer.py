import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm
import torchvision.utils as utils 
import data

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        if not args.no_test : self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
    def update_loader(self,):
        loader = data.Data(self.args)
        self.loader_train = loader.loader_train
        if not self.args.no_test : self.loader_test = loader.loader_test
    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, item in tqdm(enumerate(self.loader_train)):
            (lrs, hrs, _,) = item
            idx_scale = 3

  
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            lossF = torch.zeros(1).cuda()
            lossS = torch.zeros(1).cuda()

            estimate = torch.zeros(hrs[:,0].size()).cuda()        
            lr = torch.zeros(lrs[:,0].size()).cuda()

            for frame in range(self.args.n_frame):

                lr, lr_ = lrs[:, frame], lr#lrs[:, frame]
                hr = hrs[:, frame]
                lr, lr_, hr = self.prepare([lr, lr_, hr])
                
                

                sr, lre = self.model(lr, lr_, estimate)
                
                lossF += self.loss(lre, lr)
                lossS += self.loss(sr, hr)

                estimate = sr
                


            loss = lossF + lossS

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lrs, hrs, filename,) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hrs.nelement() == 1)

                    estimate = torch.zeros(hrs[:,0].size()).cuda()        
                    lr = torch.zeros(lrs[:,0].size()).cuda()
                    srs = torch.zeros(hrs.size())#.cuda()
                    
                    for frame in range(self.args.n_frame):
                        
                        lr, lr_ = lrs[:, frame], lr#lrs[:, frame]
                        hr = hrs[:, frame]
                        lr, lr_, hr = self.prepare([lr, lr_, hr])
                        

                        sr, lre = self.model(lr, lr_, estimate)
                        

                        estimate = sr
                        srs[0,frame].copy_(sr[0])
                        '''
                        if not no_eval:
                            lr, hr = self.prepare([lr, hr])
                        else:
                            lr = self.prepare([lr])[0]
                        
                        sr = self.model(lr, idx_scale)
                        '''
                        sr = utility.quantize(sr, self.args.rgb_range)

                    utils.save_image(torch.cat([sr, hr],-1)/255, 'patch/{}.png'.format(idx_img))

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            srs, hrs, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

