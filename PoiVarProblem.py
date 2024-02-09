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
        

        self.opts=kwargs
        # default 1
        self.p = 1

        self.param = {'D': kwargs['exact_param']}
        self.testcase = kwargs['testcase']


        self.output_transform = lambda x, u: u * x * (1 - x)
        self.dataset = None

    def D_exact(self, x):
        if self.testcase == 0:
            # constant coefficient
            return self.param['D']
        elif self.testcase == 1:
            # variable coefficient
            return 1 + 0.5 * torch.sin(2 * torch.pi * x)
        else:
            raise ValueError('Invalid testcase')
    
    def f(self, x):
        if self.testcase == 0:
            return -(torch.pi )**2 * torch.sin(torch.pi  * x)
        elif self.testcase == 1:
            v =  torch.pi**2 * torch.cos(torch.pi * x) * torch.cos(2 * torch.pi * x) - \
            torch.pi**2 * torch.sin(torch.pi * x) * (1 + 0.5 * torch.sin(2 * torch.pi * x))
            return v
        else:
            raise ValueError('Invalid testcase')

    def u_exact(self, x):
        if self.testcase == 0:
            return torch.sin(torch.pi  * x) / self.param['D']
        elif self.testcase == 1:
            return torch.sin(torch.pi  * x)
        else:
            raise ValueError('Invalid testcase')
        

    def residual(self, nn, x):
        
        x.requires_grad_(True)
        
        
        u = nn(x, None)
        D = nn.params_expand['D']
        u_x = torch.autograd.grad(u, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = torch.autograd.grad(u_x * D, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        
        res = u_xx - self.f(x)

        return res, u

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
        dataset['u_dat_test'] = self.u_exact(dataset['x_dat_test'])

        # data col-pt, for initialization use init_param, for training use exact_param
        dataset['x_dat_train'] = torch.linspace(0, 1, dsopt['N_dat_train']).view(-1, 1)

        dataset['u_dat_train'] = self.u_exact(dataset['x_dat_train'])

        # D(x) at collocation point
        dataset['D_dat_train'] = self.D_exact(dataset['x_dat_train'])
        dataset['D_dat_test'] = self.D_exact(dataset['x_dat_test'])

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
            
            self.dataset['upred_res_test'] = net(self.dataset['x_res_test'], None)
            coef = net.func_param(self.dataset['x_res_test'])
            self.dataset['func_res_test'] = coef


            self.dataset['upred_dat_test'] = net(self.dataset['x_dat_test'], None)
            coef = net.func_param(self.dataset['x_dat_test'])
            self.dataset['func_dat_test'] = coef
        
        # make prediction with different parameters
        self.prediction_variation(net)

    def prediction_variation(self, net):
        # make prediction with different parameters
        x_test = self.dataset['x_dat_test']
        deltas = [0.0, 0.1, -0.1]

        for delta in deltas:
            # replace parameter
            with torch.no_grad():
                
                u_test = net(x_test, None)
                
            key = f'upred_del{delta}_dat_test'
            self.dataset[key] = u_test

    def validate(self, nn):
        '''compute l2 error and linf error of inferred D(x)'''
        x  = self.dataset['x_dat_test']
        D = self.D_exact(x)
        with torch.no_grad():
            Dpred = nn.func_param(x)
            l2norm = torch.mean(torch.square(D - Dpred))
            linfnorm = torch.max(torch.abs(D - Dpred)) 
        
        return {'l2err': l2norm.item(), 'linferr': linfnorm.item()}

    def plot_upred(self, savedir=None):
        fig, ax = plt.subplots()
        ax.plot(self.dataset['x_dat_test'], self.dataset['u_dat_test'], label='Exact')
        ax.plot(self.dataset['x_dat_test'], self.dataset['upred_dat_test'], label='NN')
        ax.scatter(self.dataset['x_dat_train'], self.dataset['u_dat_train'], label='data')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    def plot_Dpred(self, savedir=None):
        ''' plot predicted d and exact d'''
        fig, ax = plt.subplots()
        ax.plot(self.dataset['x_dat_test'], self.dataset['D_dat_test'], label='Exact')
        ax.plot(self.dataset['x_dat_test'], self.dataset['func_dat_test'], label='NN')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_D_pred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')

    def visualize(self, savedir=None):
        '''visualize the problem'''
        self.plot_upred(savedir)
        self.plot_Dpred(savedir)
        
        
        # plot prediction variation
        # x = self.dataset['x_dat_test']
        # for delta in [0.0, 0.1, -0.1]:
        #     key = f'upred_del{delta}_dat_test'
        #     plt.plot(x, self.dataset[key], label=f'Delta {delta}')
        
        # plt.xlim(0, 1) # fix
        # plt.grid()
        # plt.legend()

        # if savedir is not None:
        #     path = os.path.join(savedir, 'fig_prediction_variation.png')
        #     plt.savefig(path, dpi=300, bbox_inches='tight')
        #     print(f'fig saved to {path}')

                
        
        


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


