#!/usr/bin/env python
# options for operator learning experiments

import sys
import json
from BaseOption import BaseOptions

from MlflowHelper import MlflowHelper


default_opts = {
    'logger_opts': {'use_mlflow':True,
                    'use_stdout':False,
                    'use_csv':False,
                    'experiment_name':'dev',
                    'run_name':'test',
                    'save_dir':'tmp'},
    'restore': '',
    'traintype': 'pretrain',
    'flags': '', 
    'device': 'cuda',
    'seed': 0,
    
    'nn_opts': {
        'depth': 4,
        'width': 64,
        'input_dim': 1,
        'output_dim': 1,
    },
    'dataset_opts': {
        'N_res_train': 101,
        'N_res_test': 101,
        'N_dat_train': 101,
        'N_dat_test': 101,

        # for heat problem
        'N_ic_train':101, # point for evaluating l2grad
        # for heat and FK problem
        'Nx':51,
        'Nt':51,
    },
    'train_opts': {
        'print_every': 20,
        'max_iter': 100000,
        'tolerance': 1e-6, # stop if loss < tolerance
        'patience': 1000,
        'lr': 1e-3,
    },
    'noise_opts':{
        'use_noise': False,
        'variance': 0.01,
        'length_scale': 0.0,
    },
}


class OptionsOpLearn(BaseOptions):
    def __init__(self):
        self.opts = default_opts
    
    def processing(self):
        ''' handle dependent options '''

        # training type 
        # pretrain: train neural operator with only data loss
        # inverse: solve inverse problem
        assert self.opts['traintype'] in {'pretrain','inverse'}, 'invalid traintype'
        




if __name__ == "__main__":
    opts = OptionsOpLearn()
    opts.parse_args(*sys.argv[1:])

    print (json.dumps(opts.opts, indent=2,sort_keys=True))