# coding: utf-8

import numpy as np
import os
import torch
import torch.nn.functional as F
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import time
import datetime
from torch.autograd import Variable
from torchvision import transforms

from train_dataloader import MyDataset
from face_model import DeepFace
import common

class Solver(object):
    def __init__(self):
        self.batch_size = 256
        self.image_size = 152
        self.n_class = 2
        self.train_path = "/home/xpp/data/VggFace2/train_face"
        self.test_path = "/home/xpp/data/VggFace2/test_face"

        self.lr = 3e-4
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.train_dataloader = MyDataset(image_path=self.train_path,
                                          batch_size=self.batch_size,
                                          image_size=self.image_size)
        self.val_dataloader = MyDataset(image_path=self.test_path,
                                        batch_size=self.batch_size,
                                        image_size=self.image_size)

        self.train_epoch = 20
        self.save_model_per_epoch = 5
        self.log_step = 20
        self.log_path = "XecpOut/logs"
        self.model_path = "XecpOut/models"
        common.create_dirs([self.log_path, self.model_path])

        self.use_tensorboard = True
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        self.pretrained_model = False
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        self.net = DeepFace()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                          self.lr, [self.beta1, self.beta2])

        if torch.cuda.is_available():
            self.net.cuda()

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        self.net.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def load_pretrained_model(self):
        self.net.load_state_dict(torch.load(os.path.join(
            self.model_path, '{}_G.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def validate(self):
        iters_all = len(self.val_dataloader) // self.batch_size
        loss = 0.0
        acc = 0.0
        for iter in range(iters_all):
            print(iter, iters_all)
            images1, images2, labels = self.val_dataloader()
            images1 = self.to_var(images1)
            images2 = self.to_var(images2)
            labels = self.to_var(labels.view(-1))
            out_cls = self.net(images1, images2)
            loss_cls = F.binary_cross_entropy_with_logits(out_cls, labels).cpu().item()
            loss += loss_cls
            pred = out_cls.cpu().detach().numpy() > 0.5
            labels = labels.cpu().detach().numpy() > 0.5
            acc += (np.equal(pred, labels).sum())
        return loss / iters_all, acc / (iters_all * self.batch_size)

    def train(self):
        # Start training
        start_time = time.time()
        # Start training
        start = 0
        if self.pretrained_model:
            start = self.pretrained_model
        for e in range(start, self.train_epoch):
            iters_per_epoch = len(self.train_dataloader) // self.batch_size
            print("iters_per_eopch:", iters_per_epoch)
            for iter in range(iters_per_epoch):
                images1, images2, labels = self.train_dataloader()
                # print(images1.size(), images2.size(), labels.size())
                images1 = self.to_var(images1)
                images2 = self.to_var(images2)
                labels = self.to_var(labels.view(-1))
                out_cls = self.net(images1, images2)
                loss_cls = F.binary_cross_entropy_with_logits(out_cls, labels)
                # Backward + Optimize
                self.reset_grad()
                loss_cls.backward()
                self.optimizer.step()
                # Logging
                loss = {}
                loss['loss_cls'] = loss_cls.item()
                # val
                if (iter + 1) == iters_per_epoch:
                    val_loss, val_acc = self.validate()

                    _log = "val_loss: {}, val_acc:{}".format(val_loss, val_acc)
                    print(_log)
                    _loss = {}
                    _loss['val_loss'] = val_loss
                    _loss['val_acc'] = val_acc
                    if self.use_tensorboard:
                        for tag, value in _loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + iter + 1)
                # Print log info
                if (iter+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.train_epoch, iter+1, iters_per_epoch)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + iter + 1)
                # Cosin learning decay
                lr = 0.5 * (1 + math.cos(iter * 3.14159 / iters_per_epoch)) * self.lr
                self.update_lr(lr)
            if e % self.save_model_per_epoch == 0:
                torch.save(self.net.state_dict(),
                       os.path.join(self.model_path, '{}_G.pth'.format(e)))

if __name__ == "__main__":
    solver = Solver()
    solver.train()
