import sys
from scipy.io import loadmat, savemat
import numpy as np
import torch

class DataSet(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def readmat(self, file_path):
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
    
    def printsummary(self):
        '''print data set
        '''
        # print name, type, and shape of each variable
        for key, value in self.items():
            shape = None
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                shape = value.shape
            print(f'{key}:\t{type(value)}\t{shape}')
    
    def save(self, file_path):
        '''save data set to .mat file
        '''
        # save data set to .mat file
        savemat(file_path, self)
    
    def to_torch(self):
        '''convert numpy array to torch tensor
        '''
        for key, value in self.items():
            self[key] = torch.tensor(value)
    

if __name__ == "__main__":
    # read mat file and print dataset
    filename  = sys.argv[1]

    # which variables to print
    vars2print = sys.argv[2:] if len(sys.argv) > 2 else None

    dataset = DataSet()
    dataset.readmat(filename)
    dataset.to_torch()
    if vars2print is None:
        dataset.printsummary()
        print(dataset)
    else:
        for var in vars2print:
            print(dataset[var])
    