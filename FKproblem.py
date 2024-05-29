#!/usr/bin/env python
# # define problems for PDE
import torch
from DataSet import DataSet
import numpy as np
from matplotlib import pyplot as plt
import os
from util import generate_grf, griddata_subsample
from BaseProblem import BaseProblem
    
class FKproblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 2 # x, t
        self.output_dim = 1
        self.opts=kwargs

        self.dataset = DataSet(kwargs['datafile'])
        self.D = self.dataset['D']
        self.RHO = self.dataset['RHO']

        self.testcase = kwargs['testcase']
        
        self.param = {}
        self.param['rD'] = self.dataset[f'rD{self.testcase}']
        self.param['rRHO'] = self.dataset[f'rRHO{self.testcase}']

        # use residual point for data loss
        self.dat_use_res = kwargs['dat_use_res']

        # ic, u(x) = 0.5*sin(pi*x)^2
        # bc, u(t,0) = 0, u(t,1) = 0
        # transform: u(x,t) = u0(x) + u_NN(x,t) * x * (1-x) * t
        self.lambda_transform = lambda X, u: (0.5 * torch.sin(np.pi * X[:,1:2]) ** 2)+ u * X[:,1:2] * (1 - X[:,1:2]) * X[:,0:1]



    def residual(self, nn, X):
        X.requires_grad_(True)

        t = X[:, 0:1]
        x = X[:, 1:2]

        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat((t,x), dim=1)

        # Forward pass through the network
        u_pred = nn(nn_input, nn.params_dict)

        # Define a tensor of ones for grad_outputs
        v = torch.ones_like(u_pred)
        
        # Compute gradients with respect to the sliced tensors
        u_t = torch.autograd.grad(u_pred, t, grad_outputs=v, create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=v, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=v, create_graph=True)[0]

        
        # Compute the right-hand side of the PDE
        rhs = nn.params_expand['rD'] * self.D * u_xx + nn.params_expand['rRHO'] * self.RHO * u_pred * (1 - u_pred)
        
        # Compute the residual
        res = u_t - rhs
        
        return res, u_pred

    def get_res_pred(self, net):
        ''' get residual and prediction'''
        res, pred = self.residual(net, self.dataset['X_res_train'])
        return res, pred
    
    def get_data_loss(self, net):
        # get data loss
        u_pred = net(self.dataset['X_dat_train'],net.params_dict)
        loss = torch.mean(torch.square(u_pred - self.dataset['u_dat_train']))
        return loss
    
    
    
    def print_info(self):
        # print parameter
        print('D = ', self.D)
        print('RHO = ', self.RHO)
        print('rD = ', self.param['rD'])
        print('rRHO = ', self.param['rRHO'])
    
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        x_dat = self.dataset['X_dat']
        x_res = self.dataset['X_res']
        
        x_dat_train = self.dataset['X_dat_train']
        x_res_train = self.dataset['X_res_train']
        
        with torch.no_grad():
            self.dataset['upred_dat'] = net(x_dat, net.params_dict)
            self.dataset['upred_res'] = net(x_res, net.params_dict)
            self.dataset['upred_dat_train'] = net(x_dat_train, net.params_dict)
            self.dataset['upred_res_train'] = net(x_res_train, net.params_dict)
        
        self.prediction_variation(net)
        
    def plot_scatter(self, X, u, fname = 'fig_scatter.png', savedir=None):
        ''' plot u vs x, color is t'''
        x = X[:,1]
        t = X[:,0]
        
        # visualize the results
        fig, ax = plt.subplots()
        
        # scatter plot, color is upred
        ax.scatter(x, u, c=t, cmap='viridis')

        if savedir is not None:
            fpath = os.path.join(savedir, fname)
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax
    
    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()        
        ax, fig = self.plot_scatter(self.dataset['X_dat'], self.dataset['upred_dat'], fname = 'fig_upred_dat.png', savedir=savedir)
        ax, fig = self.plot_scatter(self.dataset['X_res'], self.dataset['upred_res'], fname = 'fig_upred_res.png', savedir=savedir)
        ax, fig = self.plot_scatter(self.dataset['X_dat_train'], self.dataset['upred_dat_train'], fname = 'fig_upred_dat_train.png', savedir=savedir)
        ax, fig = self.plot_scatter(self.dataset['X_res_train'], self.dataset['upred_res_train'], fname = 'fig_upred_res_train.png', savedir=savedir)
        ax, fig = self.plot_scatter(self.dataset['X_dat_train'], self.dataset['u_dat_train'], fname = 'fig_u_dat_train.png', savedir=savedir)

        self.plot_upred_dat(savedir=savedir)
        self.plot_sample(savedir=savedir)
        self.plot_variation(savedir=savedir)
    

    def prediction_variation(self, net):
        # make prediction with different parameters
        # variation name = f'var_{param_name}_{delta_i}_pred'
        # only look at final time
        x_test = self.dataset['X_dat']

        deltas = [0.0, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5]
        self.dataset['deltas'] = deltas
        # copy the parameters, DO NOT modify the original parameters
        tmp_param_dict = {k: v.clone() for k, v in net.params_dict.items()}
        # go through all the trainable pde parameters
        for k in net.trainable_param:
            param_value = tmp_param_dict[k].item()
            param_name = k

            for delta_i, delta in enumerate(deltas):
                new_value = param_value + delta
                
                with torch.no_grad():
                    tmp_param_dict[param_name].data = torch.tensor([[new_value]]).to(x_test.device)
                    u_test = net(x_test, tmp_param_dict)
                    vname = f'var_{param_name}_{delta_i}_pred'
                    self.dataset[vname] = u_test

                    if hasattr(self, 'u_exact'):
                        u_exact = self.u_exact(x_test, tmp_param_dict)
                        vname = f'var_{param_name}_{delta_i}_exact'
                        self.dataset[vname] = u_exact
    
    def plot_variation(self, savedir=None):
        # go through uvar and var
            
        x = self.dataset['X_dat'][:,1]

        deltas = self.dataset['deltas']
        
        for param_name in ['rD','rRHO']:
            for delta_i, delta in enumerate(deltas):
                fig, ax = plt.subplots()
                vname = f'var_{param_name}_{delta_i}_pred'
            
                ax.plot(x, self.dataset['u_dat'], label='u')
                ax.plot(x, self.dataset[vname], label=f'u-{param_name}-{delta}')
                ax.legend(loc="best")
                
                if savedir is not None:
                    path = os.path.join(savedir, f'fig_var_{param_name}_{delta_i}.png')
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    print(f'fig saved to {path}')
                
                plt.close(fig)



    def create_dataset_from_file(self, dsopt):
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

    def plot_upred_dat(self, savedir=None):
        fig, ax = plt.subplots()
        x = self.dataset['X_dat'][:, 1]
        x_train = self.dataset['X_dat_train'][:, 1]
        ax.plot(x, self.dataset['u_dat'], label='exact')
        ax.plot(x, self.dataset['upred_dat'], label='NN')
        ax.scatter(x_train, self.dataset['u_dat_train'], label='data')
        ax.legend(loc="best")
        ax.grid()
        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred_xdat.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')

    def plot_sample(self, savedir=None):
        '''plot distribution of collocation points'''
        fig, ax = plt.subplots()
        ax.scatter(self.dataset['X_res_train'][:, 1], self.dataset['X_res_train'][:, 0], s=2.0, marker='.', label='X_res_train')
        ax.scatter(self.dataset['X_dat_train'][:, 1], self.dataset['X_dat_train'][:, 0], s=2.0, marker='s', label='X_dat_train')
        # same xlim ylim
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        
        # legend
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_Xdist.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')


if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.opts['pde_opts']['problem'] = 'fk'
    optobj.opts['pde_opts']['trainable_param'] = 'rD,rRHO'


    optobj.parse_args(*sys.argv[1:])
    
    
    device = set_device('cuda')
    set_seed(0)
    
    print(optobj.opts)

    prob = FKproblem(**optobj.opts['pde_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])
    prob.setup_dataset(optobj.opts['dataset_opts'], optobj.opts['noise_opts'])

    prob.make_prediction(pdenet)
    prob.visualize(savedir=optobj.opts['logger_opts']['save_dir'])

