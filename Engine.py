#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import os

import numpy as np
from Options import *
from util import *

from DensePoisson import *

import mlflow
from MlflowHelper import MlflowHelper
from DataSet import DataSet
from Problems import *
from lossCollection import *

from PlotHelper import *
from Logger import Logger
from Trainer import Trainer
from torchinfo import summary

class Engine:
    def __init__(self, opts=None) -> None:

        self.device = set_device()
        self.opts = opts
        self.restore_artifacts = {}
        self.logger = None
        self.dataset = {}
        self.lossCollection = {}
        self.logger = None
        self.artifact_dir = None
        self.trainer = None

        self.restore_run()
    
    def setup_logger(self):
        self.logger = Logger(self.opts['logger_opts'])

    def setup_problem(self):
        # setup pde problem
        self.pde = create_pde_problem(**(self.opts['pde_opts']),datafile=self.opts['dataset_opts']['datafile'])
        self.pde.print_info()
        

    def restore_run(self):
        # actual restore is called in setup_lossCollection, need to known collection of trainable parameters
        if self.opts['restore'] != '':
            # if is director
            if os.path.isdir(self.opts['restore']):
                opts_path = os.path.join(self.opts['restore'], 'options.json')
                restore_opts = read_json(opts_path)
                self.restore_artifacts = {fname: os.path.join(self.opts['restore'], fname) for fname in os.listdir(self.opts['restore']) if fname.endswith('.pth')}
                print('restore from directory')

            else:
                #  restore from exp_name:run_name
                restore_opts, self.restore_artifacts = load_artifact(name_str=self.opts['restore'])
                print('restore from mlflow')
        
            # udpate nn_opts from load, get network structure, catch key error
            # do not update trainable parameters, might change
            try:
                restore_opts['nn_opts']['trainable_param'] = self.opts['nn_opts']['trainable_param']
                self.opts['nn_opts'].update(restore_opts['nn_opts'])
            except KeyError:
                print('options not found, make sure the network structure is the same')
                pass
    
    def setup_network(self):
        '''setup network, get network structure if restore'''
        
        self.opts['nn_opts']['input_dim'] = self.pde.input_dim
        self.opts['nn_opts']['output_dim'] = self.pde.output_dim

        # first copy self.pde.param, which include all pde-param in network
        # then update by init_param if provided
        pde_param = self.pde.param.copy()
        init_param = self.opts['init_param']
        if init_param is not None:
            pde_param.update(init_param)

        self.net = DensePoisson(**self.opts['nn_opts'],
                                output_transform=self.pde.output_transform,
                                params_dict=pde_param)
        self.net.to(self.device)
        

    def create_dataset_from_file(self):
        dsopt = self.opts['dataset_opts']
        self.dataset = DataSet()
        self.dataset.readmat(dsopt['datafile'])
        
    
    def setup_data(self):
        '''setup data from file or from PDE
        u_dat_train are subject to noise
        '''
        if self.opts['dataset_opts']['datafile'] == '':
            # when exact pde solution is avialble, use it to create dataset
            print('create dataset from pde')
            self.dataset = create_dataset_from_pde(self.pde, self.opts['dataset_opts'])
        else:
            print('create dataset from file')
            self.create_dataset_from_file()

        # down sample training data 
        if self.opts['dataset_opts']['N_dat_train'] < self.dataset['x_dat_train'].shape[0]:
            print('downsample training data')
            self.dataset.uniform_downsample(self.opts['dataset_opts']['N_dat_train'], ['x_dat_train','u_dat_train'])

        if self.opts['noise_opts']['use_noise']:
            print('add noise to training data')
            add_noise(self.dataset, self.opts['noise_opts'])

        self.dataset.to_device(self.device)
    
    def make_prediction(self):
        self.dataset['u_res_test'] = self.net(self.dataset['x_res_test'])
        

    def setup_trainer(self):
        self.lossCollection = lossCollection(self.net, self.pde, self.dataset, self.opts['weights'])
        self.trainer = Trainer(self.opts['train_opts'], self.net, self.pde, self.dataset, self.lossCollection, self.logger)
        self.trainer.config_train(self.opts['traintype'])

        if self.opts['restore'] != '':
            self.trainer.restore(self.restore_artifacts['artifacts_dir'])
    
    def run(self):
        # training
        print_dict(self.opts)
        self.logger.log_options(self.opts)

        self.trainer.train()
        self.trainer.save()
    


if __name__ == "__main__":
    # test all component
    optobj = Options()
    optobj.parse_args(*sys.argv[1:])
    
    # set seed
    set_seed(optobj.opts['seed'])

    eng = Engine(optobj.opts)

    eng.setup_problem()
    eng.setup_network()
    eng.setup_logger()
    eng.setup_data()
    eng.setup_trainer()
    
    summary(eng.net)
    eng.run()