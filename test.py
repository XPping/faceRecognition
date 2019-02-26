# coding: utf-8

import numpy as np
import os
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import datetime
from torch.autograd import Variable
from torchvision import transforms

from face_align import alignCropFace
from face_model import DeepFace
import common



class Solver(object):
    def __init__(self):
        self.n_class = 2
        self.image_size = 152

        self.log_path = "XecpOut/logs"
        self.model_path = "XecpOut/models"

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((123.68 / 255, 116.779 / 255, 103.939 / 255),
                                 (58.393 / 255, 47.12 / 255, 57.375 / 255))
        ])

        self.build_model()

        self.pretrained_model = 20
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        self.net = DeepFace()
        if torch.cuda.is_available():
            self.net.cuda()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def load_pretrained_model(self):
        self.net.load_state_dict(torch.load(os.path.join(
            self.model_path, '{}_G.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(os.path.join(
            self.model_path, '{}_G.pth'.format(self.pretrained_model))))

    def judgeOne(self, imagepath1, imagepath2):
        face1 = alignCropFace(imagepath1)
        face2 = alignCropFace(imagepath2)

        face1 = Image.fromarray(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB))
        face1 = self.transform(face1)
        face2 = Image.fromarray(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB))
        face2 = self.transform(face2)

        face1 = self.to_var(face1)
        face2 = self.to_var(face2)
        face1 = face1.view(1, 3, self.image_size, self.image_size)
        face2 = face2.view(1, 3, self.image_size, self.image_size)
        out_cls = self.net(face1, face2)
        pred = out_cls.cpu().detach().numpy() > 0.5
        return pred[0]

if __name__ == "__main__":
    solver = Solver()
    print(solver.judgeOne(r"database/01_0001.jpg", r"database/01_0002.jpg"))
    print(solver.judgeOne(r"database/02_0001.jpg", r"database/02_0002.jpg"))
    print(solver.judgeOne(r"database/03_0001.jpg", r"database/03_0002.jpg"))
    # print(solver.judgeOne(r"database/04_0001.jpg", r"database/04_0002.jpg"))
    print(solver.judgeOne(r"database/01_0001.jpg", r"database/02_0002.jpg"))
    print(solver.judgeOne(r"database/02_0001.jpg", r"database/03_0002.jpg"))
    print(solver.judgeOne(r"database/03_0001.jpg", r"database/01_0002.jpg"))
    # print(solver.judgeOne(r"database/04_0001.jpg", r"database/01_0002.jpg"))
