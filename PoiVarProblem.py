#!/usr/bin/env python
# PoissonProblem with variable parameter
import torch
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from util import generate_grf, add_noise

from BaseProblem import BaseProblem
from DataSet import DataSet

class PoiVarProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.tag=['exact']

        self.opts=kwargs
        # default 1
        self.p = 1

        self.param = {'D': kwargs['exact_param']}
        

        self.output_transform = lambda x, u: u * x * (1 - x)
        self.dataset = None


    def residual(self, nn, x):
        def f(x):
            return -(torch.pi * self.p)**2 * torch.sin(torch.pi * self.p * x)
        x.requires_grad_(True)
        
        Dx = nn.func_param(x) # D(x)
        u_pred = nn(x, Dx)
        u_x = torch.autograd.grad(u_pred, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        
        res = Dx * u_xx - f(x)

        return res, u_pred

    def u_exact(self, x, param:dict):
        return torch.sin(torch.pi * self.p * x) / param['D']

    def print_info(self):
        # print info of pde
        # print all parameters
        print('Parameters:')
        for k,v in self.param.items():
            print(f'{k} = {v}')
        print(f'p = {self.p}')  

    def create_dataset_from_pde(self, dsopt):
        # create dataset from pde using datset option and noise option
        dataset = DataSet()

        # residual col-pt (collocation point), no need for u
        dataset['x_res_train'] = torch.linspace(0, 1, dsopt['N_res_train'] ).view(-1, 1)
        dataset['x_res_test'] = torch.linspace(0, 1, dsopt['N_res_test']).view(-1, 1)

        # data col-pt, for testing, use exact param
        dataset['x_dat_test'] = torch.linspace(0, 1, dsopt['N_dat_test']).view(-1, 1)
        dataset['u_dat_test'] = self.u_exact(dataset['x_dat_test'], self.param)

        # data col-pt, for initialization use init_param, for training use exact_param
        dataset['x_dat_train'] = torch.linspace(0, 1, dsopt['N_dat_train']).view(-1, 1)

        dataset['u_dat_train'] = self.u_exact(dataset['x_dat_train'], self.param)

        # D(x) at collocation point
        dataset['D_dat_train'] = torch.full_like(dataset['x_dat_train'], self.param['D'])
        dataset['D_res_train'] = torch.full_like(dataset['x_res_train'], self.param['D'])

        self.dataset = dataset


    def setup_dataset(self, dsopt, noise_opt):
        '''add noise to dataset'''
        self.create_dataset_from_pde(dsopt)
        if noise_opt['use_noise']:
            add_noise(self.dataset, noise_opt)
    
    def func_mse(self, net):
        '''mean square error of variable parameter'''
        x = self.dataset['x_res_train']
        y = net.func_param(x)
        return torch.mean(torch.square(y - self.dataset['D_res_train']))
    
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        with torch.no_grad():
            coef = net.func_param(self.dataset['x_res_test'])
            self.dataset['upred_res_test'] = net(self.dataset['x_res_test'], coef)
            self.dataset['func_res_test'] = coef


            coef = net.func_param(self.dataset['x_dat_test'])
            self.dataset['upred_dat_test'] = net(self.dataset['x_dat_test'], coef)
            self.dataset['func_dat_test'] = coef

    def visualize(self, savedir=None):
        '''iterat through dataset, plot x_dat_test, (var)_data_test '''
        for k, v in self.dataset.items():
            y = v
            if k.endswith('dat_test') and k!='x_dat_test':
                x = self.dataset['x_dat_test']
            elif k.endswith('res_test') and k!='x_res_test':
                x = self.dataset['x_res_test']
            else:
                continue

            
            plt.plot(x, y, label=k)
            plt.xlim(0, 1) # fix
            plt.grid()
            plt.legend()

            plt.show()
            
            if savedir is not None:
                path = os.path.join(savedir, 'fig_' + k + '.png')
                plt.savefig(path, dpi=300, bbox_inches='tight')
                print(f'fig saved to {path}')
            plt.close()

                
        
        


if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.parse_args(*sys.argv[1:])
    optobj.opts['nn_opts']['with_func'] = True
    
    
    device = set_device('cuda')
    set_seed(0)
    
    print(optobj.opts)

    prob = PoiVarProblem(**optobj.opts['pde_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])
    prob.create_dataset_from_pde(optobj.opts['dataset_opts'])


    prob.make_prediction(pdenet)
    prob.visualize(save_dir=optobj.opts['logger_opts']['save_dir'])


