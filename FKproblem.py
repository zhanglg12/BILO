# define problems for PDE
import torch
from DataSet import DataSet
import numpy as np
from matplotlib import pyplot as plt
import os
    
class FKproblem():
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 2 # x, t
        self.output_dim = 1

        self.dataset = DataSet(kwargs['datafile'])
        # get parameter from mat file
        # check empty string
        self.param = {}
        if kwargs['datafile']:
            self.param['rD'] = self.dataset['rD']
            self.param['rRHO'] = self.dataset['rRHO']
            self.D = self.dataset['D']
            self.RHO = self.dataset['RHO']
        else:
            # error
            raise ValueError('no dataset provided for FKproblem')
        
        self.use_res = kwargs['use_res']

        # ic, u(x) = 0.5*sin(pi*x)^2
        # bc, u(t,0) = 0, u(t,1) = 0
        # transform: u(x,t) = u0(x) + u_NN(x,t) * x * (1-x) * t
        self.output_transform = lambda X, u: (0.5 * torch.sin(np.pi * X[:,1:2]) ** 2)+ u * X[:,1:2] * (1 - X[:,1:2]) * X[:,0:1]

        self.dataset['X_res'].requires_grad_(True)  



    def residual(self, nn, X, param: dict):
        
        t = X[:, 0:1]
        x = X[:, 1:2]
        
        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat((t,x), dim=1)
        
        # Forward pass through the network
        u_pred = nn(nn_input)
        
        # Define a tensor of ones for grad_outputs
        v = torch.ones_like(u_pred)
        
        # Compute gradients with respect to the sliced tensors
        u_t = torch.autograd.grad(u_pred, t, grad_outputs=v, create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=v, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=v, create_graph=True)[0]

        
        # Compute the right-hand side of the PDE
        rhs = param['rD'] * self.D * u_xx + param['rRHO'] * self.RHO * u_pred * (1 - u_pred)
        
        # Compute the residual
        res = u_t - rhs
        
        return res, u_pred

    def get_res_pred(self, net):
        ''' get residual and prediction'''
        res, pred = self.residual(net, self.dataset['X_res_train'], net.params_dict)
        return res, pred
    
    def get_data_loss(self, net):
        # get data loss
        if self.use_res:
            u_pred = net(self.dataset['X_res_train'])
            loss = torch.mean(torch.square(u_pred - self.dataset['u_res_train']))
        else:
            u_pred = net(self.dataset['X_dat_train'])
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
        
        with torch.no_grad():
            self.dataset['upred_dat'] = net(x_dat)
            self.dataset['upred_res'] = net(x_res)

    def plot_scatter_pred(self, savedir=None):
        self.dataset.to_np()        
        ax, fig = self.plot_scatter(self.dataset['X_dat'], self.dataset['upred_dat'], fname = 'fig_upred_dat.png', savedir=savedir)
        ax, fig = self.plot_scatter(self.dataset['X_res'], self.dataset['upred_res'], fname = 'fig_upred_res.png', savedir=savedir)
        
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

    def setup_dataset(self, ds_opts, noise_opts=None):
        ''' downsample for training'''
        
        # data loss
        ndat_train = min(ds_opts['N_dat_train'], self.dataset['X_dat'].shape[0])
        vars = self.dataset.filter('_dat')
        self.dataset.subsample_unif_astrain(ndat_train, vars)
        print('unif downsample ', vars, ' to ', ndat_train)

        # res loss
        nres_train = min(ds_opts['N_res_train'], self.dataset['X_res'].shape[0])
        vars = self.dataset.filter('_res')
        self.dataset.subsample_unif_astrain(nres_train, vars)
        print('unif downsample ', vars, ' to ', nres_train)


    