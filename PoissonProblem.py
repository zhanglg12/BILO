# define problems for PDE
import torch
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from util import generate_grf, add_noise

from BaseProblem import BaseProblem
from DataSet import DataSet

class PoissonProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.tag=['exact']
        
        self.opts=kwargs
        # default 1
        self.p = 1

        self.param = {'D': 2.0}
        if kwargs.get('exact_param') is not None:
            self.param['D'] = kwargs['exact_param']['D']

        self.output_transform = lambda x, u: u * x * (1 - x)


    def residual(self, nn, x, param:dict):
        def f(x):
            return -(torch.pi * self.p)**2 * torch.sin(torch.pi * self.p * x)
        x.requires_grad_(True)
        u_pred = nn(x)
        u_x = torch.autograd.grad(u_pred, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = param['D'] * u_xx - f(x)

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

        self.dataset = dataset


    def setup_dataset(self, dsopt, noise_opt):
        '''add noise to dataset'''
        self.create_dataset_from_pde(dsopt)
        if noise_opt['use_noise']:
            add_noise(self.dataset, noise_opt)