# coding: utf-8

import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import cv2


class MyDataset(Dataset):
    def __init__(self, image_path, batch_size, image_size=152):
        self.image_path = image_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((123.68 / 255, 116.779 / 255, 103.939 / 255), (58.393 / 255, 47.12 / 255, 57.375 / 255))
        ])
        print('Start preprocessing dataset..!')
        self.preprocess()
        print('Finished preprocessing dataset..!')

    def preprocess(self):
        samples = {}
        person = {}

        dir1 = os.listdir(self.image_path)
        for i, d1 in enumerate(dir1):
            if d1 not in person:
                person[d1] = i
            samples[i] = []
            first_dir = os.path.join(self.image_path, d1)
            dir2 = os.listdir(first_dir)
            for d2 in dir2:
                samples[i].append(os.path.join(first_dir, d2))
            samples[i] = np.array(samples[i])
        # Save person id correspond to label
        with open("person_id.txt", 'w') as fw:
            for key in person.keys():
                fw.write(key+'\t'+str(person[key])+'\n')

        self.samples = samples
        self.shuffle()

        self.iter = 0

    def shuffle(self):
        self.samples1 = []
        self.samples2 = []
        self.labels = []

        # Same person pairs
        for key in self.samples.keys():
            perm = np.random.permutation(len(self.samples[key]))
            self.samples[key] = self.samples[key][perm]
            for i in range(0, len(self.samples[key])-1, 2):
                self.samples1.append(self.samples[key][i])
                self.samples2.append(self.samples[key][i+1])
                self.labels.append(1.0)
        same_person_pairs = len(self.samples1)
        # Not same person pairs
        samples1 = []
        labels1 = []
        for key in self.samples.keys():
            for i in range(len(self.samples[key])):
                samples1.append(self.samples[key][i])
                labels1.append(int(key))
        samples1 = np.array(samples1)
        labels1 = np.array(labels1)
        samples2 = samples1[:]
        labels2 = labels1[:]
        perm = np.random.permutation(len(samples2))
        samples2 = samples2[perm]
        labels2 = labels2[perm]

        for i in range(len(samples1)):
            if labels1[i] != labels2[i]:
                self.samples1.append(samples1[i])
                self.samples2.append(samples2[i])
                self.labels.append(0.)
        self.samples1 = np.array(self.samples1)
        self.samples2 = np.array(self.samples2)
        self.labels = np.array(self.labels)
        perm = np.random.permutation(len(self.samples1))
        self.samples1 = self.samples1[perm]
        self.samples2 = self.samples2[perm]
        self.labels = self.labels[perm]

        print("Same person pairs: {}, not same person pairs: {}".format(same_person_pairs, len(self.samples1)-same_person_pairs))

    def __call__(self, *args, **kwargs):
        iter_step = len(self.samples1) // self.batch_size
        if self.iter >= iter_step - 1:
            self.shuffle()
            self.iter = 0
        samples1 = self.samples1[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
        samples2 = self.samples2[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
        labels = self.labels[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
        self.iter += 1

        images1 = []
        images2 = []

        for s in samples1:
            image = cv2.imread(s)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.transform(image)
            images1.append(image)
        for s in samples2:
            image = cv2.imread(s)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.transform(image)
            images2.append(image)

        _labels = []
        for i in range(len(labels)):
            _labels.append(torch.FloatTensor([labels[i]]))
        images1 = torch.stack(images1, 0)
        images2 = torch.stack(images2, 0)
        _labels = torch.stack(_labels, 0)

        return images1, images2, _labels

    def __len__(self):
        return len(self.samples1)
