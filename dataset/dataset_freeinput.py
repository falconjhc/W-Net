import numpy as np
import tensorflow as tf
import os

import sys
sys.path.append('..')
from utilities.utils import image_show
import random as rnd
import time
import multiprocessing as multi_thread
print_separator = "#################################################################"
from tensorflow.python.client import device_lib
import copy as cpy



class Dataset(object):
    def __init__(self):
        a=1