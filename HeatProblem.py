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
from DenseNet import DenseNet, ParamFunction


class HeatDenseNet(DenseNet):
    ''' override the embedding function of DenseNet'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # override the embedding function, also enforce dirichlet boundary condition
        self.func_param = ParamFunction(depth=4, width=32, output_transform=lambda x, u: u * x * (1.0 - x))

    def embedding(self, x, params_dict=None):
        # override the embedding function
        # x: (batch, 2), second dimension is the x coordinate
        # fourier feature embedding
        xcoord  = x[:, 1:2]
        if self.fourier:
            x = torch.sin(2 * torch.pi * self.fflayer(x))
        x = self.input_layer(x)

        # have to evaluate self.func_param(xcoord) inside the network
        # otherwise self.func_param is not in the computation graph
        assert self.with_func is True, "with_func=True to use HeatDenseNet"
        assert params_dict is None, "have to eval f(x) inside the network"
        
        if self.with_param :
            # go through each parameter and do embedding
            if self.with_func is False:
                for name, param in params_dict.items():
                        # expand the parameter to the same size as x
                        self.params_expand[name] = param.expand(x.shape[0], -1)
                        scalar_param_expanded = self.params_expand[name] # (batch, 1)
                        param_embedding = self.param_embeddings[name](scalar_param_expanded)
                        x += param_embedding
            else:
                # evaluted func_param at xcoord, then do embedding
                # CAUTION: assuming only learning one function, therefore only self.func_param intead of a dict
                for name in self.params_dict.keys():
                    param_vector = self.func_param(xcoord)
                    self.params_expand[name] = param_vector
                    param_embedding = self.param_embeddings[name](param_vector)
                    x += param_embedding
    
        else:
            if self.with_func is False:
                # for vanilla version, no parameter embedding
                # copy params_dict to params_expand
                for name, param in self.params_dict.items():
                    self.params_expand[name] = params_dict[name]
            else:
                for name in self.params_dict.keys():
                    self.params_expand[name] =  self.func_param(xcoord)
        return x
    
    def output_transform(self, x, u):
        # override the output transform attribute of DenseNet
        u0 = self.func_param(x[:, 1:2])
        return u * x[:, 1:2] * (1 - x[:, 1:2]) * x[:, 0:1] + u0


            
class HeatProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 1
        self.opts=kwargs
 
        self.testcase = kwargs['testcase']
        self.D = kwargs['D']

        # placeholder for parameter, only name is used
        self.param = {'u0': 0.0}

        # do not pass transformation to network
        # make transformation in residual loss and data loss
        self.lambda_transform = lambda x, u: u

        self.dataset = None
    
    def no_grad_evaluate(self, nn, X):
        '''evaluate the network without gradient'''
        with torch.no_grad():
            t = X[:, 0:1]
            x = X[:, 1:2]

            nn_out = nn(X, None)
            u0 = nn.params_expand['u0']
            u = nn_out * x * (1 - x) * t + u0
        return u, u0

    def u_exact(self, t, x):
        if self.testcase == 0:
            # exact solution E^(-D Pi^2 t) Sin[Pi x]
            return np.sin(np.pi  * x) * np.exp(-np.pi**2 * self.D * t)
        elif self.testcase == 1:
            # inifite serie
            # C = -16/pi^3
            # sum_n=1^inf C (-1+(-1)^n) Sin[n pi x] E^(-D n^2 pi^2 t)/n^3
            # truncate to 100 terms
            C = -16/np.pi**3
            u = 0
            for n in range(1, 101):
                u += C * (-1 + (-1)**n) * np.sin(n * np.pi * x) * np.exp(-self.D * n**2 * np.pi**2 * t) / n**3
            return u
        else:
            raise ValueError('Invalid testcase')
        
    def u0_exact(self, x):
        if self.testcase == 0:
            # initial condition Sin[Pi x]
            return np.sin(np.pi  * x)
        elif self.testcase == 1:
            # initial condition 4 x (1 - x)
            return 4 * x * (1 - x)
        else:
            raise ValueError('Invalid testcase')
    
    def residual(self, nn, X_in):
        
        t = X_in[:, 0:1]
        x = X_in[:, 1:2]
        
        # Concatenate sliced tensors to form the input for the network
        X = torch.cat((t,x), dim=1)

        nn_out = nn(X, None)
        u = nn_out * x * (1 - x) * t + nn.params_expand['u0']
        
        u_t = torch.autograd.grad(u, t,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_x = torch.autograd.grad(u, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = torch.autograd.grad(u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        
        res = u_t - self.D * u_xx

        return res, u

    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        pde_param = self.param.copy()
        init_param = self.opts['init_param']
        if init_param is not None:
            pde_param.update(init_param)

        net = HeatDenseNet(**kwargs,
                            lambda_transform=self.lambda_transform,
                            params_dict=pde_param,
                            trainable_param = self.opts['trainable_param'])
        return net

    def print_info(self):
        # print info of pde
        # print all parameters
        print('Parameters:')
        for k,v in self.param.items():
            print(f'{k} = {v}')
    
    def create_dataset_from_pde(self, dsopt):
        # create dataset from pde using datset option and noise option
        dataset = DataSet()

        Nt = dsopt['Nt']
        Nx = dsopt['Nx']

        dataset['Nt'] = Nt
        dataset['Nx'] = Nx
        
        t = np.linspace(0, 1, Nt).reshape(-1, 1)
        x = np.linspace(0, 1, Nx).reshape(-1, 1)
        # T,X are Nx by Nt, first dimension is x, second dimension is t
        # after flattern, t increase and repeat, x repeat and increase
        T, X = np.meshgrid(t, x)

        # reshape to (Nt*Nx, 2)
        dataset['X_res'] = np.column_stack((T.flatten().reshape(-1, 1), X.flatten().reshape(-1, 1)))
        u_res = self.u_exact(T, X)
        dataset['u_res'] = u_res.flatten().reshape(-1, 1)
        
        # X_dat is (all 1, x)
        dataset['X_dat'] = np.column_stack((np.ones((Nx,1)), x))
        u_dat = self.u_exact(1.0, x)
        dataset['u_dat'] = u_dat.reshape(-1, 1)
 
        # x_ic is x coord only
        dataset['x_ic'] = x
        dataset['u0_exact_ic'] = self.u0_exact(x)
        

        self.dataset = dataset
        self.dataset.to_torch()
        self.dataset.printsummary()


    def setup_dataset(self, dsopt, noise_opt = None):
        '''add noise to dataset'''
        self.create_dataset_from_pde(dsopt)
        
    
    def func_mse(self, net):
        '''mean square error of variable parameter'''
        x = self.dataset['x_ic']
        y = net.func_param(x)
        return torch.mean(torch.square(y - self.dataset['u0_exact_ic']))
    
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        with torch.no_grad():
            self.dataset['upred_res'] = net(self.dataset['X_res'], None)
            self.dataset['upred_dat'] = net(self.dataset['X_dat'], None)
            self.dataset['u0_pred_ic'] = net.func_param(self.dataset['x_ic'])
        

    def validate(self, nn):
        '''compute l2 error and linf error of inferred D(x)'''
        x  = self.dataset['x_ic']
        u0_exact = self.u0_exact(x)
        with torch.no_grad():
            u0_pred = nn.func_param(x)
            err = u0_exact - u0_pred
            l2norm = torch.mean(torch.square(err))
            linfnorm = torch.max(torch.abs(err)) 
        
        return {'l2err': l2norm.item(), 'linferr': linfnorm.item()}

    def plot_upred_dat(self, savedir=None):
        fig, ax = plt.subplots()
        x = self.dataset['X_dat'][:, 1]
        ax.plot(x, self.dataset['u_dat'], label='exact')
        ax.plot(x, self.dataset['upred_dat'], label='NN')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred_xdat.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    def plot_upred_res_meshgrid(self, savedir=None):
        # plot u at X_res, 
        fig, ax = plt.subplots(1, 3)
        
        u = self.dataset['u_res']
        u_pred = self.dataset['upred_res']
        err = u - u_pred

        # reshape to 2D
        u = u.reshape(self.dataset['Nt'], self.dataset['Nx'])
        u_pred = u_pred.reshape(self.dataset['Nt'], self.dataset['Nx'])
        # 2D plot
        ax[0].imshow(u_pred , cmap='viridis', extent=[0, 1, 0, 1], origin='lower', vmin=0, vmax=1)
        ax[0].set_title('NN')

        ax[1].imshow(u , cmap='viridis', extent=[0, 1, 0, 1], origin='lower', vmin=0, vmax=1)
        ax[1].set_title('Exact')

        ax[2].imshow(err, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
        ax[2].set_title('Error')

        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred_grid.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    def plot_upred_res(self, savedir=None):
        # plot u at X_res,         
        u = self.dataset['u_res']
        u_pred = self.dataset['upred_res']
        err = u - u_pred
        
        # uniformly spaced interger between 0 and Nt
        N = 5
        tidx = np.linspace(0, self.dataset['Nt']-1, N, dtype=int)

        # reshape to 2D
        u = u.reshape(self.dataset['Nt'], self.dataset['Nx'])
        u_pred = u_pred.reshape(self.dataset['Nt'], self.dataset['Nx'])

        fig, ax = plt.subplots()
        for i in tidx:
            # plot u at each t
            ax.plot(u[:, i], label=f'Exact t={i/self.dataset["Nt"]:.2f}')
            ax.plot(u_pred[:, i], label=f'NN t={i/self.dataset["Nt"]:.2f}', linestyle='--')
            
        

        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred_xres.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    

    
    def plot_ic_pred(self, savedir=None):
        ''' plot predicted d and exact d'''
        fig, ax = plt.subplots()
        ax.plot(self.dataset['x_ic'], self.dataset['u0_pred_ic'], label='NN')
        ax.plot(self.dataset['x_ic'], self.dataset['u0_exact_ic'], label='Exact')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_ic_pred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')

    def visualize(self, savedir=None):
        '''visualize the problem'''
        self.plot_upred_res_meshgrid(savedir=savedir)
        self.plot_upred_res(savedir=savedir)
        self.plot_upred_dat(savedir=savedir)
        self.plot_ic_pred(savedir=savedir)
                            
        
        


if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.opts['pde_opts']['problem'] = 'HeatProblem'
    optobj.opts['nn_opts']['with_func'] = True
    optobj.opts['pde_opts']['trainable_param'] = 'u0'


    optobj.parse_args(*sys.argv[1:])
    
    
    device = set_device('cuda')
    set_seed(0)
    
    print(optobj.opts)

    prob = HeatProblem(**optobj.opts['pde_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])
    prob.setup_dataset(optobj.opts['dataset_opts'])

    prob.make_prediction(pdenet)
    prob.visualize(savedir=optobj.opts['logger_opts']['save_dir'])


