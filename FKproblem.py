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

        # tag for plotting, ode: plot component, 2d: plot traj, exact: have exact solution
        self.tag = ['pde','2d']
        
        # ic, u(x) = 0.5*sin(pi*x)^2
        # bc, u(t,0) = 0, u(t,1) = 0
        # transform: u(x,t) = u0(x) + u_NN(x,t) * x * (1-x) * t
        self.output_transform = lambda X, u: (0.5 * torch.sin(np.pi * X[:,0:1]) ** 2)+ u * X[:,0:1] * (1 - X[:,0:1]) * X[:,1:2]



    def residual(self, nn, X, param: dict):
        
        x = X[:, 0:1]
        t = X[:, 1:2]
        
        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat((x, t), dim=1)
        
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

    
    def print_info(self):
        # print parameter
        print('D = ', self.D)
        print('RHO = ', self.RHO)
        print('rD = ', self.param['rD'])
        print('rRHO = ', self.param['rRHO'])
    


    def plot_pred_seq(self, dataset, net, savedir=None):
        # though this is a 2d problem, we can plot the prediction in sequence
        
        x_dat_test = dataset['x_dat_test']
        u_dat_test = dataset['u_dat_test']
        
        with torch.no_grad():
            upred = net(x_dat_test)

        # move to cpu
        x_dat_test = x_dat_test.cpu().detach().numpy()
        upred = upred.cpu().detach().numpy()
        u_test = u_dat_test.cpu().detach().numpy()
        
        # visualize the results
        fig, ax = plt.subplots()
        
        # plot the prediction in sequence
        ax.plot(upred, label=f'pred')
        ax.plot(u_test, label=f'train')

        if savedir is not None:
            fpath = os.path.join(savedir, f'fig_pred_seq.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax

    def plot_scatter(self, dataset, net, savedir=None):
        x_dat_test = dataset['x_dat_test']
        u_dat_test = dataset['u_dat_test']
        
        with torch.no_grad():
            upred = net(x_dat_test)

        # move to cpu
        x_dat_test = x_dat_test.cpu().detach().numpy()
        upred = upred.cpu().detach().numpy()
        u_test = u_dat_test.cpu().detach().numpy()
        
        # visualize the results
        fig, ax = plt.subplots()
        
        # scatter plot, color is upred
        ax.scatter(x_dat_test[:,0], x_dat_test[:,1], c=u_test, cmap='viridis')

        if savedir is not None:
            fpath = os.path.join(savedir, f'fig_scatter.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax
    
    def visualize(self, dataset, net, savedir=None):
        # visualize the results
        self.plot_pred_seq(dataset, net, savedir=savedir)
        self.plot_scatter(dataset, net, savedir=savedir)



    