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
        return string
    
    def save(self, file_path):
        '''save data set to .mat file
        '''
        # save data set to .mat file
        print(f'save dataset to {file_path}')
        self.to_np()
        savemat(file_path, self)
    
    def to_torch(self):
        '''convert numpy array to torch tensor
        skip string
        '''
        print('convert dataset to torch')
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                self[key] = torch.tensor(value,dtype=torch.float32)
    
    def to_np(self, d = None):
        '''convert tensor to cpu
        '''
        if d is None:
            d = self
            print('move dataset to cpu')

        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                d[key] = value.cpu().detach().numpy()
            # if dictionary, recursively convert
            elif isinstance(value, dict):
                self.to_np(d = value)
                
    
    def to_device(self, device):
        print(f'move dataset to {device}')
        for key, value in self.items():
            try:
                self[key] = value.to(device)
            except AttributeError:
                # skip non-tensor
                # print(f'skip {key}')
                pass
    
    def subsample_evenly_astrain(self, n, vars):
        '''uniformly downsample data set in the first dimension
        n is final number of samples
        '''
        N = self[vars[0]].shape[0]
        step = (N-1)//(n-1)
        for var in vars:
            self[var+'_train'] = self[var][::step,:]
    
    def filter(self, substr):
        ''' return list of key that contains substr
        '''
        return [key for key in self.keys() if substr in key]
    
    def subsample_firstn_astrain(self, n, vars):
        '''subsample first n row for training. 
        add _train suffix to variable name
        '''
        for var in vars:
            self[var+'_train'] = self[var][:n]
        
    def subsample_unif_astrain(self, n, vars):
        '''subsample n row uniformly for training. 
        add _train suffix to variable name
        '''
        N = self[vars[0]].shape[0]
        idx = np.random.choice(N, n, replace=False)
        for var in vars:
            self[var+'_train'] = self[var][idx]
            
    def remove(self, keys):
        '''remove variables that contains substr
        '''
        for key in keys:
            self.pop(key, None)
            print(f'remove {key}')
    

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
    