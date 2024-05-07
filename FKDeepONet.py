#!/usr/bin/env python
import os
import numpy as np
import torch
from matplotlib import pyplot as plt

from DataSet import DataSet 
from DeepONet import DeepONet, OpData
from BaseOperator import BaseOperator
from util import griddata_subsample, generate_grf

class FKOperatorLearning(BaseOperator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = 2
        self.output_dim = 1
        self.param_dim = 2
        self.lambda_transform = lambda X, u: (0.5 * torch.sin(np.pi * X[:,1:2]) ** 2)+ u * X[:,1:2] * (1 - X[:,1:2]) * X[:,0:1]
        self.testcase = kwargs['testcase']
    
    def get_metrics(self, nn:DeepONet):
        # take pde_param, tensor of trainable parameters
        # return dictionary of metrics
        return {'rD': nn.pde_param[0,0].item(), 'rRHO': nn.pde_param[0,1].item()}

    def setup_network(self, **kwargs):
        deeponet = DeepONet(param_dim=self.param_dim, X_dim=self.input_dim, output_dim=self.output_dim, **kwargs,
         lambda_transform=self.lambda_transform)

        D0 = 1.0
        rho0 = 1.0        
        deeponet.pde_param = torch.nn.Parameter(torch.tensor([D0, rho0], dtype=torch.float32).reshape(1,2).to('cuda'))

        return deeponet

    def create_dataset_from_file(self, dsopt):
        # create data for inverse problem
        dataset = self.dataset
        
        uname = f'u{self.testcase}'

        u = dataset[uname]
        gt = dataset['gt']
        gx = dataset['gx']
        Nt_full, Nx_full = u.shape
        
        # downsample size
        Nt = dsopt['Nt']
        Nx = dsopt['Nx']
        dataset['Nt'] = Nt
        dataset['Nx'] = Nx

        # collect X and u from final time
        X_dat = np.column_stack((gt[-1, :].reshape(-1, 1), gx[-1, :].reshape(-1, 1)))
        u_dat = u[-1, :].reshape(-1, 1)
        dataset['X_dat'] = X_dat
        dataset['u_dat'] = u_dat
        # downsample for training
        idx = np.linspace(0, Nx_full-1, dsopt['N_dat_train'], dtype=int)
        dataset['X_dat_train'] = X_dat[idx, :]
        dataset['u_dat_train'] = u_dat[idx, :]


        # collect X and u from all time, for residual loss
        dataset['X_res'] = np.column_stack((gt.reshape(-1, 1), gx.reshape(-1, 1)))
        dataset['u_res'] = u.reshape(-1, 1)

        # for training, downsample griddata and vectorize
        gt, gx, u = griddata_subsample(gt, gx, u, Nt, Nx)
        dataset['X_res_train'] = np.column_stack((gt.reshape(-1, 1), gx.reshape(-1, 1)))
        dataset['u_res_train'] = u.reshape(-1, 1)
        
        # remove redundant data
        for i in range(10):
            if i != self.testcase:
                dataset.pop(f'u{i}',None)
                dataset.pop(f'ic{i}',None)
        
        dataset.printsummary()
        

    def setup_dataset(self, ds_opts, noise_opts=None):
        ''' downsample for training'''
        
        self.create_dataset_from_file(ds_opts)
        self.dataset.to_torch()

        if noise_opts['use_noise']:
            print('add noise to training data')
            x = self.dataset['X_dat_train'][:,1:2]
            noise = torch.zeros_like(self.dataset['u_dat_train'])
    
            tmp = generate_grf(x, noise_opts['variance'], noise_opts['length_scale'])
            noise[:,0] = tmp.squeeze()

            self.dataset['noise'] = noise
            self.dataset['u_dat_train'] = self.dataset['u_dat_train'] + self.dataset['noise']    

    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.plot_upred_dat(savedir=savedir)

    def make_prediction_pretrain(self, deeponet:DeepONet):
        P = self.dataset['P']
        U_pred = deeponet(P, self.dataset['X'])

        self.pred_dataset = DataSet()
        self.pred_dataset['U'] = U_pred

        err = U_pred - self.dataset['U']


    def make_prediction_inverse(self, deeponet: DeepONet):
        # make prediction at original X_dat and X_res
        x_dat = self.dataset['X_dat']
        x_res = self.dataset['X_res']
        
        x_dat_train = self.dataset['X_dat_train']
        x_res_train = self.dataset['X_res_train']
        
        with torch.no_grad():
            self.dataset['upred_dat'] = deeponet(deeponet.pde_param, x_dat)
            self.dataset['upred_res'] = deeponet(deeponet.pde_param, x_res)
            self.dataset['upred_dat_train'] = deeponet(deeponet.pde_param, x_dat_train)
            self.dataset['upred_res_train'] = deeponet(deeponet.pde_param, x_res_train)
    
    def plot_upred_dat(self, savedir=None):
        fig, ax = plt.subplots()
        x = self.dataset['X_dat'][:, 1]
        x_train = self.dataset['X_dat_train'][:, 1]
        
        ax.plot(x, np.squeeze(self.dataset['u_dat']), label='exact')
        ax.plot(x, np.squeeze(self.dataset['upred_dat']), label='NN')
        ax.scatter(x_train, np.squeeze(self.dataset['u_dat_train']), label='data')

        ax.legend(loc="best")
        ax.grid()
        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred_xdat.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')


if __name__ == "__main__":
    import sys
    # test the OpDataSet class
    filename  = sys.argv[1]
    fkoperator = FKOperatorLearning(datafile=filename)

    fkoperator.init(batch_size=1000)
    fkoperator.train(max_iter=1000, print_every=100, save_every=5000)

    fkoperator.save_data('pred.mat')


    