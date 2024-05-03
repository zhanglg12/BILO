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
    'datafile': '',
    'nn_opts': {
        'branch_depth': 4,
        'trunck_depth': 4,
        'width': 64,
    },
    'train_opts': {
        'print_every': 20,
        'save_every':5000,
        'max_iter': 100000,
        'tolerance': 1e-6, # stop if loss < tolerance
        'patience': 1000,
        'lr': 1e-3,
        'batch_size': 1000,
        'split': 0.8,
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
        
        if self.opts['flags'] != '':
            self.opts['flags'] = self.opts['flags'].split(',')
            assert all([flag in ['small','local'] for flag in self.opts['flags']]), 'invalid flag'
        else:
            self.opts['flags'] = []

        if 'small' in self.opts['flags']:
            # use small network for testing
            self.opts['nn_opts']['depth'] = 4
            self.opts['nn_opts']['width'] = 2
            self.opts['train_opts']['max_iter'] = 10
            self.opts['train_opts']['print_every'] = 1
        
        if 'local' in self.opts['flags']:
            # use local logger
            self.opts['logger_opts']['use_mlflow'] = False
            self.opts['logger_opts']['use_stdout'] = True
            self.opts['logger_opts']['use_csv'] = False



if __name__ == "__main__":
    opts = OptionsOpLearn()
    opts.parse_args(*sys.argv[1:])

    print (json.dumps(opts.opts, indent=2,sort_keys=True))