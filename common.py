# coding: utf-8

import os
import numpy as np


def create_dirs(names):
    for name in names:
        if not os.path.exists(name):
            os.makedirs(name)
