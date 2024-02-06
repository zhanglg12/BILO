#!/usr/bin/env python
# define problems for PDE
from Options import *
from util import *
import torch
from DataSet import DataSet
from matplotlib import pyplot as plt
import os
from DensePoisson import DensePoisson
from BaseProblem import BaseProblem

def sumcol(A):
    # sum along column
    return torch.sum(A, dim=1, keepdim=True)



class GBMproblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.dataset = DataSet(kwargs['datafile'])
        # get parameter from mat file
        # check empty string
        self.param = {}
        self.opts = kwargs

        # GBM specific options
        self.whichdata = kwargs['whichdata']

        
        self.xdim = int(self.dataset['xdim'])
        self.dim = self.xdim + 1 # add time dimension
        self.input_dim = self.dim
        self.output_dim = 1


        
        # inititalize parameters 
        self.param['rD'] = 1.0
        self.param['rRHO'] = 1.0
        
        self.DW = self.dataset['DW']
        self.RHO = self.dataset['RHO']
        
        self.x0 = self.dataset['x0char']
        self.L = self.dataset['L']



        self.output_transform = torch.nn.Module()
        self.output_transform.register_buffer('x0',  torch.tensor(self.x0))
        self.output_transform.register_buffer('L', torch.tensor(self.L))
        self.output_transform.forward = lambda X, u: self.ic(X, self.output_transform.x0, self.output_transform.L) + u * X[:,0:1]

        self.dataset['X_res'].requires_grad_(True)  
        

    def ic(self, X, x0, L):
        # initial condition
        r2 = sumcol(torch.square((X[:, 1:self.dim] - x0)*L)) # this is in pixel scale, unit mm, 
        return 0.1*torch.exp(-0.1*r2)

    def residual(self, nn, X, phi, P, gradPphi):
        
        # Get the number of dimensions
        n = X.shape[0]

        # split each column of X into a separate tensor of size (n, 1)
        vars = [X[:, d:d+1] for d in range(self.dim)]
        
        
        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat(vars, dim=1)
       
        # Forward pass through the network
        u = nn(nn_input, nn.params_dict)
        # Define a tensor of ones for grad_outputs
        v = torch.ones_like(u)
        
        # Compute gradients with respect to the sliced tensors
        u_t = torch.autograd.grad(u, vars[0], grad_outputs=v, create_graph=True)[0]

        # n by d matrix
        u_x = torch.zeros(n, self.xdim, device=X.device)
        u_xx = torch.zeros(n, self.xdim, device=X.device)

        for d in range(0, self.xdim):
            u_x[:,d:d+1] = torch.autograd.grad(u, vars[d+1], grad_outputs=v, create_graph=True)[0]
            u_xx[:,d:d+1] = torch.autograd.grad(u_x[:,d:d+1], vars[d+1], grad_outputs=v, create_graph=True)[0]
        
        prof = nn.params_expand['rRHO'] * self.RHO * phi * u * ( 1 - u)
        diff = nn.params_expand['rD'] * self.DW * (P * phi * sumcol(u_xx) + self.L * sumcol(gradPphi * u_x))
        res = phi * u_t - (prof + diff)
        return res, u
    
    
    def get_res_pred(self, net):
        # get residual and prediction
        res, u_pred = self.residual(net, self.dataset['X_res_train'], self.dataset['phi_res_train'], self.dataset['P_res_train'], self.dataset['gradPphi_res_train'])
        return res, u_pred

    def get_data_loss(self, net):
        # get data loss
        u_pred = net(self.dataset['X_dat_train'], net.params_dict)
        loss = torch.mean(torch.square((u_pred - self.dataset['u_dat_train'])*self.dataset['phi_dat_train']))
        return loss
    
    def get_bc_loss(self, net):
        # get dirichlet boundary condition loss
        u_pred = net(self.dataset['X_bc_train'], net.params_dict)
        loss = torch.mean(torch.square((u_pred - self.dataset['zero_bc_train'])))
        return loss
    
    def print_info(self):
        pass
    
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

            
    def plot_scatter(self, X, u, fname = 'fig_scatter.png', savedir=None):

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
    
    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()        
        ax, fig = self.plot_scatter(self.dataset['X_dat'], self.dataset['upred_dat'], fname = 'fig_upred_dat.png', savedir=savedir)
        ax, fig = self.plot_scatter(self.dataset['X_res'], self.dataset['upred_res'], fname = 'fig_upred_res.png', savedir=savedir)
        ax, fig = self.plot_scatter(self.dataset['X_dat_train'], self.dataset['upred_dat_train'], fname = 'fig_upred_dat_train.png', savedir=savedir)
        ax, fig = self.plot_scatter(self.dataset['X_res_train'], self.dataset['upred_res_train'], fname = 'fig_upred_res_train.png', savedir=savedir)
    
    def setup_dataset(self, ds_opts, noise_opts=None):
        ''' downsample for training'''
        
        # data loss
        ndat_train = min(ds_opts['N_dat_train'], self.dataset['X_dat'].shape[0])
        vars = self.dataset.filter('_dat')
        self.dataset.subsample_firstn_astrain(ndat_train, vars)
        print('downsample ', vars, ' to ', ndat_train)

        # res loss
        nres_train = min(ds_opts['N_res_train'], self.dataset['X_res'].shape[0])
        vars = self.dataset.filter('_res')
        self.dataset.subsample_firstn_astrain(nres_train, vars)
        print('downsample ', vars, ' to ', nres_train)

        # bc loss
        n = min(ds_opts['N_bc_train'], self.dataset['X_bc'].shape[0])
        vars = self.dataset.filter('_bc')
        self.dataset.subsample_firstn_astrain(n, vars)
        print('downsample ', vars, ' to ', n)


        # which data is (uchar|ugt)_(res|dat)
        key = self.whichdata + '_train'
        self.dataset['u_dat_train'] = self.dataset[key]
        
        # where to evaluated the data loss, _res or _dat
        if 'res' in self.whichdata:
            self.dataset['X_dat_train'] = self.dataset['X_res_train']
            self.dataset['phi_dat_train'] = self.dataset['phi_res_train']
            print('use res data for data loss, change X_dat_train and phi_dat_train')
        
        
        
            
        


        
        
        



# simple test of the pde
# creat a network, check if residual can be computed correctly
if __name__ == "__main__":

    
    optobj = Options()
    optobj.parse_args(*sys.argv[1:])
    
    device = set_device('cuda')
    set_seed(0)
    
    prob = GBMproblem(**optobj.opts['pde_opts'])
    prob.print_info()
    prob.setup_dataset(optobj.opts['dataset_opts'])

    net = prob.setup_network(**optobj.opts['nn_opts'])

    res, u_pred = prob.residual(net, prob.dataset['X_res'], prob.dataset['phi_res'], prob.dataset['P_res'], prob.dataset['gradPphi_res'])

    prob.make_prediction(net)
    prob.visualize(savedir=optobj.opts['logger_opts']['save_dir'])
    

    # print 2 norm of res
    print('res = ',torch.norm(res))