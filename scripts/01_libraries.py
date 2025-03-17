import torch as torch
import torch.nn as nn         # classes and functions for NN building
import torch.optim as optim   # optimization algorithms for training
from torch.utils.data import Dataset, DataLoader
import torchvision            # pre-trained models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder # for batching and loading data
import timm                                  # contains the models

import matplotlib.pyplot as plt              # for data viz
import numpy as np
