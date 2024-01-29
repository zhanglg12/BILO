#!/usr/bin/env python
# define problems for PDE
from Options import *
from util import *
import torch
from DataSet import DataSet
from matplotlib import pyplot as plt
import os
from DensePoisson import DensePoisson

def sumcol(A):
    # sum along column
    return torch.sum(A, dim=1, keepdim=True)



class GBMproblem():
    def __init__(self, **kwargs):
        super().__init__()
        
        self.dataset = DataSet(kwargs['datafile'])
        # get parameter from mat file
        # check empty string
        self.param = {}
        
        self.xdim = int(self.dataset['xdim'])
        self.dim = self.xdim + 1 # add time dimension
        self.input_dim = self.dim
        self.output_dim = 1

        self.param['rD'] = self.dataset['rDe']
        self.param['rRHO'] = self.dataset['rRHOe']
        
        self.DW = self.dataset['DW']
        self.RHO = self.dataset['RHO']
        
        self.x0 = self.dataset['x0char']
        self.L = self.dataset['L']


        self.output_transform = torch.nn.Module()
        self.output_transform.register_buffer('x0',  torch.tensor(self.x0))
        self.output_transform.register_buffer('L', torch.tensor(self.L))
        self.output_transform.forward = lambda X, u: self.ic(X, self.output_transform.x0, self.output_transform.L) + u * X[:,0:1]

        

    def ic(self, X, x0, L):
        # initial condition
        r2 = sumcol(torch.square((X[:, 1:self.dim] - x0)*L)) # this is in pixel scale, unit mm, 
        return 0.1*torch.exp(-0.1*r2)


    def residual(self, nn, X, phi, P, gradPphi, param: dict):
        
        # Get the number of dimensions
        n = X.shape[0]

        # split each column of X into a separate tensor of size (n, 1)
        vars = [X[:, d:d+1] for d in range(self.dim)]
        
        
        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat(vars, dim=1)
        
        # Forward pass through the network
        u = nn(nn_input)
        
        # Define a tensor of ones for grad_outputs
        v = torch.ones_like(u)
        
        # Compute gradients with respect to the sliced tensors
        u_t = torch.autograd.grad(u, vars[0], grad_outputs=v, create_graph=True)[0]

        # n by d matrix
        u_x = torch.zeros(n, self.xdim, device=X.device)
        u_xx = torch.zeros(n, self.xdim, device=X.device)

        # import pdb; pdb.set_trace()
        for d in range(0, self.xdim):
            u_x[:,d:d+1] = torch.autograd.grad(u, vars[d+1], grad_outputs=v, create_graph=True)[0]
            u_xx[:,d:d+1] = torch.autograd.grad(u_x[:,d:d+1], vars[d+1], grad_outputs=v, create_graph=True)[0]
        
        prof = param['rRHO'] * self.dataset['RHO'] * phi * u * ( 1 - u)
        diff = param['rD'] * self.dataset['DW'] * (P * phi * sumcol(u_xx) + self.dataset['L'] * sumcol(gradPphi * u_x))
        res = phi * u_t - (prof + diff)
        return res, u

    
    def print_info(self):
        pass
    

    def plot_scatter_pred(self, dataset, net, savedir=None):
        x_dat = dataset['X_dat']
        x_res = dataset['X_res']
        
        with torch.no_grad():
            upred_xdat = net(x_dat)
            upred_xres = net(x_res)

        ax, fig = self.plot_scatter(x_dat, upred_xdat, fname = 'fig_upred_xdat.png', savedir=savedir)

        ax, fig = self.plot_scatter(x_res, upred_xres, fname = 'fig_upred_xres.png', savedir=savedir)
        
    


    
    def plot_scatter(self, X, u, fname = 'fig_scatter.png', savedir=None):

        # convert to numpy if tensor
        if torch.is_tensor(X):
            X = X.cpu().detach().numpy()
        if torch.is_tensor(u):
            u = u.cpu().detach().numpy()
        
        x = X[:,1:]
        r = np.linalg.norm(x,axis=1)
        t = X[:,0]
        
        # visualize the results
        fig, ax = plt.subplots()
        
        # scatter plot, color is upred
        ax.scatter(r, u, c=t, cmap='viridis')

        if savedir is not None:
            fpath = os.path.join(savedir, fname)
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax
    
    def visualize(self, dataset, net, savedir=None):
        # visualize the results
        
        self.plot_scatter_pred(dataset, net, savedir=savedir)


# simple test of the pde
# creat a network, check if residual can be computed correctly
if __name__ == "__main__":

    
    optobj = Options()
    optobj.parse_args(*sys.argv[1:])
    

    device = set_device('cuda')
    set_seed(0)
    
    prob = GBMproblem(datafile=optobj.opts['dataset_opts']['datafile'])
    prob.print_info()

    optobj.opts['nn_opts']['input_dim'] = prob.input_dim
    optobj.opts['nn_opts']['output_dim'] = prob.output_dim

    net = DensePoisson(**optobj.opts['nn_opts'],
                output_transform=prob.output_transform, 
                params_dict=prob.param).to(device)
    
    ds = prob.dataset
    ds.to_device(device)

    ds['X_res'].requires_grad_(True)
    res, u_pred = prob.residual(net, ds['X_res'], ds['phi_res'], ds['P_res'], ds['gradPphi_res'], net.params_dict)

    prob.visualize(ds, net, savedir=optobj.opts['logger_opts']['save_dir'])
    

    # print 2 norm of res
    print('res = ',torch.norm(res))