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
        self.pde = create_pde_problem(**(self.opts['pde_opts']))
    
    def setup_network(self):
        # setup network, get network structure if restore
        # actual restore is called in setup_lossCollection, need to known collection of trainable parameters
        if self.opts['restore'] != '':
            # if is director
            if os.path.isdir(self.opts['restore']):
                opts_path = os.path.join(self.opts['restore'], 'options.json')
                opts = read_json(opts_path)
                self.restore_artifacts = {fname: os.path.join(self.opts['restore'], fname) for fname in os.listdir(self.opts['restore']) if fname.endswith('.pth')}

            else:
                #  restore from exp_name:run_name
                opts, self.restore_artifacts = load_artifact(name_str=self.opts['restore'])
        
        # udpate options from load, get network structure, catch key error
        try:
            self.opts['nn_opts'].update(opts['nn_opts'])
        except KeyError:
            print('options not found, make sure the network structure is the same')
            pass

        self.net = DensePoisson(**(self.opts['nn_opts']),params_dict=self.pde.init_param)
        self.net.to(self.device)
        self.net.params_dict.requires_grad_(True)

    
    def setup_data(self):
        dsopt = self.opts['dataset_opts']
        self.dataset = DataSet()
        xtmp = torch.linspace(0, 1, dsopt['N_res_train'] ).view(-1, 1)
        
        self.dataset['x_res_train'] = xtmp.to(self.device)
        self.dataset['x_res_train'].requires_grad_(True)

        self.dataset['x_res_test'] = torch.linspace(0, 1, dsopt['N_res_test']).view(-1, 1).to(self.device)

        # generate data, might be noisy
        if self.opts['traintype']=="init" or self.opts['traintype']=="forward":
            # for init, use D_init, no noise
            self.dataset['u_res_train'] = self.pde.u_exact(self.dataset['x_res_train'], self.pde.init_param)
        else:
            # for basci/inverse, use D_exact
            self.dataset['u_res_train'] = self.pde.u_exact(self.dataset['x_res_train'], self.pde.exact_param)

        if self.opts['noise_opts']['use_noise']:
            self.dataset['noise'] = generate_grf(xtmp, self.opts['noise_opts']['variance'],self.opts['noise_opts']['length_scale'])
            self.dataset['u_res_train'] = self.dataset['u_res_train'] + self.dataset['noise'].to(self.device)
    
    def make_prediction(self):
        self.dataset['u_res_test'] = self.net(self.dataset['x_res_test'])
        

    def setup_lossCollection(self):

        if self.opts['traintype'] == 'basic':
            # basic method of solving inverse problem, optimize weight and parameter
            # no resgrad, residual loss and data loss only, 
            loss_pde_opts = {'weights':{'res':self.opts['weights']['res'],'data':self.opts['weights']['data']}}
            self.lossCollection['basic'] = lossCollection(self.net, self.pde, self.dataset, list(self.net.parameters()), optim.Adam, loss_pde_opts)
            
        
        elif self.opts['traintype'] == 'forward':
            # fast forward solve using resdual, or together with data loss
            loss_pde_opts = {'weights':{'res':self.opts['weights']['res'],'data':self.opts['weights']['data']}}
            self.lossCollection['basic'] = lossCollection(self.net, self.pde, self.dataset, self.net.param_net, optim.Adam, loss_pde_opts)
        
        elif self.opts['traintype'] == 'init':
            # use all 3 losses
            loss_pde_opts = {'weights':{'res':self.opts['weights']['res'],
            'resgrad':self.opts['weights']['resgrad'],
            'data':self.opts['weights']['data'],
            'paramgrad':self.opts['weights']['paramgrad']}}

            self.lossCollection['pde'] = lossCollection(self.net, self.pde, self.dataset, self.net.param_net, optim.Adam, loss_pde_opts)

        elif self.opts['traintype'] == 'inverse':
            loss_pde_opts = {'weights':{'res':self.opts['weights']['res'],'resgrad':self.opts['weights']['resgrad'],'data':self.opts['weights']['data']}}
            self.lossCollection['pde'] = lossCollection(self.net, self.pde, self.dataset, self.net.param_net, optim.Adam, loss_pde_opts)

            loss_data_opts = {'weights':{'data':self.opts['weights']['data']}}
            self.lossCollection['data'] = lossCollection(self.net, self.pde, self.dataset, self.net.param_pde, optim.Adam, loss_data_opts)
        else:
            raise ValueError(f'train type {self.opts["traintype"]} not supported')
        

        if self.opts['restore']!= '':
            self.restore()

        return 



    def run(self):

        print_dict(self.opts)
        
        self.trainer = Trainer(self.opts['train_opts'], self.net, self.pde, self.dataset, self.lossCollection, self.logger)
        self.trainer.train()
        self.trainer.save(self.logger.get_dir())
        # save options
        savedict(self.opts, os.path.join(self.logger.get_dir(), 'options.json'))


    def restore(self):
        # self.restore_artifacts is dictionary of file_name: file_path of the artifacts to be restored

        # restore network structure
        self.net.load_state_dict(torch.load(self.restore_artifacts['net.pth']))
        print(f'net loaded from {self.restore_artifacts["net.pth"]}')

        # restore optimizer
        for lossObjName in self.lossCollection:
            lossObj = self.lossCollection[lossObjName]
            
            optim_fname = f"optimizer_{lossObjName}.pth"
            
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

    # eng.dataset.to_device(eng.device)

    # ph = PlotHelper(eng.pde, eng.dataset, yessave=True, save_dir=eng.logger.get_dir())
    # ph.plot_prediction(eng.net)
    # ph.plot_variation(eng.net)

    # save command to file
    f = open("commands.txt", "a")
    f.write(' '.join(sys.argv))
    f.write('\n')
    f.close()