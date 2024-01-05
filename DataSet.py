#!/usr/bin/env python

import sys
from scipy.io import loadmat, savemat
import numpy as np
import torch

class DataSet(dict):
    '''data set class
    interface between .mat file and python
    access data set as dictionary
    '''
    def __init__(self, *args, **kwargs):
        # if only one argument, assume it is a file path, remove 
        if len(args) == 1:
            file_path = args[0]
            self.readmat(file_path)
        else:
            super().__init__(*args, **kwargs)


    def readmat(self, file_path, as_torch=True):
        # load data from .mat file, skip meta data
        data = loadmat(file_path, mat_dtype=True)

        for key, value in data.items():
            
            if key.startswith("__"):
                # skip meta data
                continue
            if isinstance(value,np.ndarray):

                if value.size == 1:
                    # if singleton, get number
                    value = value.item()
                
                self[key] = value
        
        # otherwise it is a numpy array
        if as_torch:
            self.to_torch()
    
    def printsummary(self):
        '''print data set
        '''
        # print name, type, and shape of each variable
        for key, value in self.items():
            shape = None
            typename = type(value).__name__
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                shape = value.shape
                
            print(f'{key}:\t{typename}\t{shape}')
    
    def __str__(self):
        # return variable, type, and shape as string
        string = ''
        for key, value in self.items():
            shape = None
            typename = type(value).__name__ 
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                shape = value.shape
                
            string += f'{key}:\t{typename}\t{shape}\n'
    
    def save(self, file_path):
        '''save data set to .mat file
        '''
        # save data set to .mat file
        print(f'save dataset to {file_path}')
        self.to_cpu()
        savemat(file_path, self)
    
    def to_torch(self):
        '''convert numpy array to torch tensor
        skip string
        '''
        print('convert dataset to torch')
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                self[key] = torch.tensor(value,dtype=torch.float32)
    
    def to_cpu(self):
        '''convert tensor to cpu
        '''
        print('move dataset to cpu')
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.cpu().detach()
    
    def to_device(self, device):
        print(f'move dataset to {device}')
        for key, value in self.items():
            try:
                self[key] = value.to(device)
            except AttributeError:
                # skip non-tensor
                # print(f'skip {key}')
                pass
    
    def uniform_downsample(self, n, vars):
        '''uniformly downsample data set in the first dimension
        n is final number of samples
        '''
        N = self[vars[0]].shape[0]
        step = N//n
        for var in vars:
            self[var] = self[var][::step]
            
            
    

if __name__ == "__main__":
    # read mat file and print dataset
    filename  = sys.argv[1]

    # which variables to print
    vars2print = sys.argv[2:] if len(sys.argv) > 2 else None

    dataset = DataSet(filename)
    
    dataset.to_torch()
    if vars2print is None:
        dataset.printsummary()
    else:
        for var in vars2print:
            print(dataset[var])
    