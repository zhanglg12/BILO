import torch
import torch.nn as nn
import torch.optim as optim

import os
from DensePoisson import *

from util import *
device = set_device()

default_opts = {
    'model_dir': 'model',
    'traintype': 'vanilla',
    'nn_opts': {
        'depth': 3,
        'width': 64,
        'use_resnet': False,
        'basic': False,
        'init_D': 1.0,
        'p': 1
    },
    'train_opts': {
        'max_iter': 100000,
        'lr': 0.001,
        'tolerance': 1e-3,
        'print_every': 100,
    },
    'Dexact': 2.0,
    'noise_opts':{
        'use_noise': False,
        'variance': 0.01,
        'length_scale': 0.0,
    }
    
}

class Engine:
    def __init__(self, opts) -> None:

        self.opts = opts

        self.net = DensePoisson(**(self.opts['nn_opts']))
        

        self.dataset = {}
    
    def setup_data(self):
        xtmp = torch.linspace(0, 1, 20).view(-1, 1)
        self.dataset['x_train_res'] = xtmp.to(device)
        self.dataset['x_train_res'].requires_grad_(True)

        # generate data, might be noisy
        self.dataset['u_data'] = self.net.u_exact(self.dataset['x_train_res'], self.opts['Dexact'])
        if self.opts['noise_opts']['use_noise']:
            self.dataset['noise'] = generate_grf(xtmp, self.opts['noise_opts']['variance'],self.opts['noise_opts']['length_scale'])
            self.dataset['u_data'] = self.dataset['u_data'] + self.dataset['noise'].to(device)

    def solve(self):
        self.net.to(device)
        
        # print options 
        print(json.dumps(self.opts, indent=2,cls=MyEncoder,sort_keys=True))
        

        if self.opts["traintype"] == "vanilla":
            self.optimizer_full = optim.Adam(self.net.parameters(), lr = self.opts['train_opts']['lr'])  # Only D
            train_network_vanilla(self.net, self.optimizer_full, self.dataset, self.opts['train_opts'])
        
        elif self.opts["traintype"] == "init" or self.opts["traintype"] == "inverse":
            self.optimizer_net = optim.Adam(self.net.param_net, lr = self.opts['train_opts']['lr'])  # Exclude D
        
            if self.opts["traintype"] == "init":
                train_network_init(self.net, self.optimizer_net, self.dataset, self.opts['train_opts'])
            elif self.opts["traintype"] == "inverse":
                self.optimizer_D = optim.Adam(self.net.parameters(), lr = self.opts['train_opts']['lr'])
                train_network_inverse(self.net, self.optimizer_net, self.optimizer_D, self.dataset, self.opts['train_opts'])
        
        else:
            raise ValueError("traintype not recognized")    
