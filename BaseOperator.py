#!/usr/bin/env python
import os
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np

from DataSet import DataSet



class BaseOperator(ABC):
    # operator learning
    def __init__(self, **kwargs):
        self.input_dim = None
        self.output_dim = None
        self.param_dim = None

        
        self.dataset = DataSet(kwargs['datafile'])
        self.lambda_transform = None

    def regularization_loss(self, nn):
        return 0.0
        
    
