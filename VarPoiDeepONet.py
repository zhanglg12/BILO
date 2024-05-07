#!/usr/bin/env python
import os
import numpy as np
import torch
from matplotlib import pyplot as plt

from DataSet import DataSet 
from DeepONet import DeepONet, OpData
from BaseOperator import BaseOperator
from util import griddata_subsample, generate_grf, add_noise, uniform_subsample_with_endpoint

class VarPoiDeepONet(BaseOperator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = 1
        self.output_dim = 1
        self.param_dim = kwargs['param_dim']
        self.lambda_transform = lambda x, u: u * x * (1.0 - x)
        self.testcase = kwargs['testcase']
        
        # mask on the D tensor
        self.mask = torch.ones(1, self.param_dim, dtype=torch.float32)
        self.mask[0,-1] = 0.0
    
    def get_metrics(self, nn:DeepONet):
        # take pde_param, tensor of trainable parameters
        # return dictionary of metrics
        x  = self.dataset['x_dat']
        
        D = self.dataset['D_dat'] * nn.mask
        Dpred = nn.pde_param * nn.mask

        with torch.no_grad():
            l2norm = torch.mean(torch.square(D - Dpred))
            linfnorm = torch.max(torch.abs(D - Dpred)) 
        return {'l2err': l2norm.item(), 'linferr': linfnorm.item()}

    def setup_network(self, **kwargs):

        deeponet = DeepONet(X_dim=self.input_dim, output_dim=self.output_dim, param_dim=self.param_dim,
         lambda_transform=self.lambda_transform, mask = self.mask, **kwargs)

        # create tensor of all 1s, size 1-by-param_dim for initial guess
        t = torch.ones(1, self.param_dim, dtype=torch.float32)
        deeponet.pde_param = torch.nn.Parameter(t)

        return deeponet


    def make_prediction_pretrain(self, deeponet:DeepONet):
        P = self.dataset['P']
        U_pred = deeponet(P, self.dataset['X'])

        self.pred_dataset = DataSet()
        self.pred_dataset['U'] = U_pred

    def create_dataset_from_file(self, dsopt):
        '''create dataset from file'''
        assert self.dataset is not None, 'datafile provide, dataset should not be None'
        uname = f'u{self.testcase}'
        dname = f'd{self.testcase}'

    
        self.dataset['x_dat'] = self.dataset['x']
        self.dataset['u_dat'] = self.dataset[uname]
        self.dataset['D_dat'] = self.dataset[dname]


        self.dataset.subsample_evenly_astrain(dsopt['N_dat_train'], ['x_dat', 'u_dat'])
        self.dataset.subsample_evenly_astrain(self.param_dim, ['D_dat'])
    
    def get_inverse_data(self):
        '''return inverse data'''
        U = self.dataset['u_dat_train']
        U = torch.reshape(U, (1, -1))
        X = self.dataset['x_dat_train']

        return X, U

    def setup_dataset(self, dsopt, noise_opt):
        '''add noise to dataset'''
        self.create_dataset_from_file(dsopt)

        if noise_opt['use_noise']:
            add_noise(self.dataset, noise_opt)
    
    def make_prediction_inverse(self, deeponet:DeepONet):
        '''make prediction for inverse problem'''
        with torch.no_grad():

            # operator evaluate with NN D
            upred_dat = deeponet(deeponet.pde_param, self.dataset['x_dat'])
            self.dataset['upred_dat'] = upred_dat.reshape(-1, 1)

            # operator evaluate with GT D
            D = self.dataset['D_dat_train'].reshape(1, -1)
            upred_dat = deeponet(D, self.dataset['x_dat'])
            self.dataset['upred_gt_dat'] = upred_dat.reshape(-1, 1)

            # predicted D
            D_pred = deeponet.pde_param
            D_pred[0,0] = 1.0
            D_pred[0,-1] = 1.0
            self.dataset['func_dat'] = D_pred.reshape(-1, 1)



    def plot_upred(self, savedir=None):
        fig, ax = plt.subplots()
        ax.plot(self.dataset['x_dat'], self.dataset['u_dat'], label='Exact')
        ax.plot(self.dataset['x_dat'], self.dataset['upred_dat'], label='NN')
        ax.plot(self.dataset['x_dat'], self.dataset['upred_gt_dat'], label='NN_gt')
        ax.scatter(self.dataset['x_dat_train'], self.dataset['u_dat_train'], label='data')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    def plot_Dpred(self, savedir=None):
        ''' plot predicted d and exact d'''
        n = self.dataset['func_dat'].shape[0]
        x = uniform_subsample_with_endpoint(self.dataset['x_dat_train'], n)

        fig, ax = plt.subplots()
        ax.plot(x, self.dataset['D_dat_train'], label='Exact')
        ax.plot(x, self.dataset['func_dat'] , label='NN')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_D_pred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')

    def visualize(self, savedir=None):
        '''visualize the problem'''
        self.plot_upred(savedir)
        self.plot_Dpred(savedir)


    