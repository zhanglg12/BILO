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

            else:
                #  restore from exp_name:run_name
                restore_opts, self.restore_artifacts = load_artifact(name_str=self.opts['restore'])
        
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
        

    def setup_lossCollection(self):
        

        which_optim = self.opts['optimizer']
        optim_opts = self.opts[f'{which_optim}_opts']

        if self.opts['traintype'] == 'basic':
            # basic method of solving inverse problem, optimize weight and parameter
            # no resgrad, residual loss and data loss only, 
            loss_weights ={'res':self.opts['weights']['res'],'data':self.opts['weights']['data']}
            self.lossCollection['basic'] = lossCollection(self.net, self.pde, self.dataset, self.net.param_all, which_optim, optim_opts, loss_weights)
            
        
        elif self.opts['traintype'] == 'forward':
            # fast forward solve using resdual, or together with data loss
            loss_weights ={'res':self.opts['weights']['res'],'data':self.opts['weights']['data']}
            self.lossCollection['basic'] = lossCollection(self.net, self.pde, self.dataset, self.net.param_net, which_optim, optim_opts, loss_weights)
        
        elif self.opts['traintype'] == 'init':
            # use all 3 losses
            loss_weights = {'res':self.opts['weights']['res'],
            'resgrad':self.opts['weights']['resgrad'],
            'data':self.opts['weights']['data'],
            'paramgrad':self.opts['weights']['paramgrad']}

            self.lossCollection['pde'] = lossCollection(self.net, self.pde, self.dataset, self.net.param_net, which_optim, optim_opts, loss_weights)

        elif self.opts['traintype'] == 'inverse':
            
            loss_pde_weights ={'res':self.opts['weights']['res'],'resgrad':self.opts['weights']['resgrad']}
            self.lossCollection['pde'] = lossCollection(self.net, self.pde, self.dataset, self.net.param_net, which_optim, self.opts[f'adam_pde_opts'], loss_pde_weights)

            loss_data_weights ={'data':self.opts['weights']['data']}
            # self.lossCollection['data'] = lossCollection(self.net, self.pde, self.dataset, self.net.param_pde, which_optim, self.opts[f'adam_data_opts'], loss_data_weights)
            self.lossCollection['data'] = lossCollection(self.net, self.pde, self.dataset, self.net.param_pde, 'lbfgs', {'lr':1e-2}, loss_data_weights)
        else:
            raise ValueError(f'train type {self.opts["traintype"]} not supported')
        

        if self.opts['restore']!= '':
            self.restore()

        return 



    def run(self):

        print_dict(self.opts)
        self.logger.log_options(self.opts)

        self.trainer = Trainer(self.opts['train_opts'], self.net, self.pde, self.dataset, self.lossCollection, self.logger)
        self.trainer.train()
        self.trainer.save(self.logger.get_dir())
        # save options
        


    def restore(self):
        # self.restore_artifacts is dictionary of file_name: file_path of the artifacts to be restored

        # restore network structure
        self.net.load_state_dict(torch.load(self.restore_artifacts['net.pth']))
        print(f'net loaded from {self.restore_artifacts["net.pth"]}')

        # restore optimizer
        for lossObjName in self.lossCollection:
            lossObj = self.lossCollection[lossObjName]
            
            optimizer_name = lossObj.optimizer_name
            optim_fname = f"optimizer_{lossObjName}_{optimizer_name}.pth"
            
            if optim_fname in self.restore_artifacts:
                optim_path = self.restore_artifacts[optim_fname]
                lossObj.optimizer.load_state_dict(torch.load(optim_path))
                print(f'optimizer {optim_fname} loaded from {optim_path}')
            else:
                print(f'optimizer {optim_fname} not found, use default optimizer')



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

    eng.dataset.to_device(eng.device)
    eng.setup_lossCollection()
    
    summary(eng.net)

    eng.run()

    eng.dataset.to_device(eng.device)