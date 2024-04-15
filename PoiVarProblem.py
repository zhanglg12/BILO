#!/usr/bin/env python
# PoissonProblem with variable parameter
import torch
import torch.nn as nn
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from util import generate_grf, add_noise

from BaseProblem import BaseProblem
from DataSet import DataSet

from DenseNet import DenseNet, ParamFunction

from torchreparam import ReparamModule

GLOBTEST = False

class PoiDenseNet(DenseNet):
    ''' override the embedding function of DenseNet'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        fdepth = kwargs['fdepth']
        fwidth = kwargs['fwidth']
        activation = kwargs['activation']
        output_activation = kwargs['output_activation']
        fsiren = kwargs['fsiren']
        
        
        # override the embedding function, also enforce dirichlet boundary condition
        
        if GLOBTEST:
            self.func_param = ParamFunction(fdepth=1, fwidth=1, 
            activation='id',output_activation='id')
            # set bias = 1, weight = 0
            self.func_param.layers[0].bias.data.fill_(1.0)
            self.func_param.layers[0].weight.data.fill_(0.0)
            self.func_param.layers[0].weight.requires_grad = False
            
            # For testing, PoiDenseNet is linear
            self.act = nn.Identity()
            self.output_layer = nn.Identity()
        else:
            self.func_param = ParamFunction(fdepth=fdepth, fwidth=fwidth,
                                        fsiren=fsiren,
                                        activation=activation, output_activation=output_activation,
                                        output_transform=lambda x, u: u * x * (1.0 - x) + 1.0 )

        self.collect_trainable_param()
        self.D_eval = None
        
        

    def setup_embedding_layers(self, in_features=None):

        self.param_embeddings = nn.ModuleDict({'D': nn.Linear(1, self.width, bias=False)})

        if GLOBTEST:
            # set weight = 1, identity 
            self.param_embeddings['D'].weight.data.fill_(1.0)
            self.input_layer.weight.data.fill_(1.0)
            self.input_layer.bias.data.fill_(0.0)
            self.lambda_transform = lambda x, u: u**2
            
        # set requires_grad to False
        for embedding_weights in self.param_embeddings.parameters():
            embedding_weights.requires_grad = False
        
        # fix embedding layer set embedding to identity
        # self.param_embeddings['D'].weight.data.fill_(1.0)


    def embed_x(self, x):
        '''embed x to the input layer'''
        if self.fourier:
            x_embed = torch.sin(2 * torch.pi * self.fflayer(x))
        x_embed = self.input_layer(x)

        return x_embed
    
    def embedding(self, x):
        # override the embedding function
        
        # have to evaluate self.func_param(xcoord) inside the network
        # otherwise self.func_param is not in the computation graph
    
        x_embed = self.embed_x(x)

        if self.with_param:
            
            param_vector = self.func_param(x) #(n, 1) D at x_res_train
            self.D_eval = param_vector
            self.params_expand['D'] = self.D_eval
            y_embed = self.param_embeddings['D'](self.D_eval)
            
            x_embed += y_embed
        else:
            # if u(x), not function of unkown
            self.params_expand['D'] =self.func_param(x)
            self.D_eval = self.params_expand['D']

        return x_embed


        
    
    def embedding_to_u(self, X):
        # X is the embedded input, linear combination of the "features"
        Xtmp = self.act(X)
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden_output = hidden_layer(Xtmp)
            if self.use_resnet:
                hidden_output += Xtmp  # ResNet connection
            hidden_output = self.act(hidden_output)
            Xtmp = hidden_output
        
        u = self.output_layer(Xtmp)
        
        return u

    def forward(self, x):
        
        X = self.embedding(x)
        
        u = self.embedding_to_u(X)

        u = self.output_transform(x, u)
        return u
    
    def variation(self, x, z):
        '''variation of u w.r.t D
        Let z be custom function, u = u(x, z)
        '''
        x_embed = self.embed_x(x)
        z_embed = self.param_embeddings['D'](z)
        X = x_embed + z_embed

        u = self.embedding_to_u(X)
        u = self.output_transform(x, u)
        return u
    


class PoiVarProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.opts=kwargs

        self.param = {'D': kwargs['D']}
        self.testcase = kwargs['testcase']

        self.lambda_transform = lambda x, u: u * x * (1.0 - x)

        self.dataset = None
        if kwargs['datafile']:
            self.dataset = DataSet(kwargs['datafile'])

    def u_exact(self, x):
        if self.testcase == 0:
            # different D
            return torch.sin(torch.pi  * x) / self.param['D']
        elif self.testcase == 1:
            # same u as testcase 0 D = 1, change forcing
            return torch.sin(torch.pi  * x)
        elif self.testcase == 2:
            # different D, same forcing
            return 2 * torch.log(torch.cos(torch.pi * x/2.0)+torch.sin(torch.pi * x/2.0))
        elif self.testcase == 3:
            # https://www.sciencedirect.com/science/article/pii/S0377042718306344
            # u(x) = x^4 if x<0.5, 1/2(x^4 + 1/16) if x>=0.5
            # subtract 17/32 x for dirichlet bc
            return torch.where(x < 0.5, x**4, 0.5 * (x**4 + 1/16)) -  17.0/32.0 * x
        elif self.testcase == 4:
            return -x + x**4
        else:
            raise ValueError('Invalid testcase')
        
    def D_exact(self, x):
        if self.testcase == 0:
            # constant coefficient
            return self.param['D'] * torch.ones_like(x)
        elif self.testcase == 1:
            # variable coefficient
            return 1 +  0.5*torch.sin(2 * torch.pi * x)
        elif self.testcase == 2:
            return 1.0 + torch.sin(torch.pi * x)
        elif self.testcase == 3:
            return torch.where(x < 0.5, 1.0, 2.0)
        elif self.testcase == 4:
            return torch.ones_like(x)
        else:
            raise ValueError('Invalid testcase')
    
    def f_exact(self, x):
        if self.testcase == 0:
            return -(torch.pi )**2 * torch.sin(torch.pi * x)
        elif self.testcase == 1:
            
            v =  torch.pi**2 * torch.cos(torch.pi * x) * torch.cos(2 * torch.pi * x) - \
            torch.pi**2 * torch.sin(torch.pi * x) * (1 + 0.5*torch.sin(2 * torch.pi * x))
            return v
        elif self.testcase == 2:
            # same as testcase 0, different D
            return - (torch.pi )**2 * torch.sin(torch.pi * x)
        elif self.testcase == 3 or self.testcase == 4:
            return 12 * x**2
        else:
            raise ValueError('Invalid testcase')

    

    def residual(self, nn, x):
        
        x.requires_grad_(True)
        
        u = nn(x)

        D = nn.D_eval
        
        u_x = torch.autograd.grad(u, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        dxDux = torch.autograd.grad(D*u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = dxDux - self.dataset['f_res_train']
        
        # used for checking computation here
        # if GLOBTEST:
            # when taking derivative of r = (Du')' - f w.r.t D
            # r = D' u' + D u'' - f, note that u is function of D also
            # dr/dD = D' d/dD (u') + u'' + D d/dD(u'')

            # u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
            # dx = torch.autograd.grad(D, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(x))[0]
            # dz_u_xx = torch.autograd.grad(u_xx, nn.params_expand['D'], create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_xx))[0]
            # dz_u_x = torch.autograd.grad(u_x, nn.params_expand['D'], create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
            # dz_u = torch.autograd.grad(u, nn.params_expand['D'], create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]

            # tmp = dx * dz_u_x + u_xx + D * dz_u_xx

            # dres = torch.autograd.grad(res, nn.params_expand['D'], grad_outputs=torch.ones_like(res),
            #     create_graph=True, retain_graph=True,allow_unused=True)[0]
            
            # tmp is the same as dres
            # import pdb; pdb. set_trace()
                    
        return res, u

    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        pde_param = self.param.copy()
        init_param = self.opts['init_param']
        if init_param is not None:
            pde_param.update(init_param)

        net = PoiDenseNet(**kwargs,
                        lambda_transform=self.lambda_transform,
                        params_dict= self.param,
                        trainable_param = self.opts['trainable_param'])
        net.setup_embedding_layers()
        return net

    def print_info(self):
        # print info of pde
        # print all parameters
        print('Parameters:')
        for k,v in self.param.items():
            print(f'{k} = {v}')

    def get_res_pred(self, net):
        ''' get residual and prediction'''
        res, pred = self.residual(net, self.dataset['x_res_train'])
        return res, pred
    
    def get_l2grad(self, net):
        # estimate l2 norm of u, 1/N \sum u^2
        
        x = self.dataset['x_res_train']
        D = net.func_param(x)
        D_x = torch.autograd.grad(D, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(D))[0]
        return torch.mean(torch.square(D_x))

    def get_l1grad(self, net):
        # estimate l1 norm of u, 1/N \sum |u|
        
        x = self.dataset['x_res_train']
        D = net.func_param(x)
        D_x = torch.autograd.grad(D, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(D))[0]
        return torch.mean(torch.abs(D_x))
    
    def get_data_loss(self, net):
        # get data loss
        u_pred = net(self.dataset['x_dat_train'])
        loss = torch.mean(torch.square(u_pred - self.dataset['u_dat_train']))        
        return loss

    def get_l2norm(self, net):
        # estimate l2 norm of u, 1/N \sum u^2
        u_pred = net(self.dataset['x_res_train'])
        loss = torch.mean(torch.square(u_pred))
        return loss

    

    

    def create_dataset_from_pde(self, dsopt):
        # create dataset from pde using datset option and noise option
        assert self.dataset is None, 'datafile not provide, dataset should be None'
        dataset = DataSet()

        # residual col-pt (collocation point), no need for u
        dataset['x_res'] = torch.linspace(0, 1, dsopt['N_res_test']).view(-1, 1)

        # data col-pt, for testing, use exact param
        dataset['x_dat'] = torch.linspace(0, 1, dsopt['N_dat_test']).view(-1, 1)
        dataset['u_dat'] = self.u_exact(dataset['x_dat'])

        # D(x) at collocation point
        dataset['D_res'] = self.D_exact(dataset['x_res'])
        dataset['f_res'] = self.f_exact(dataset['x_res'])
        dataset['D_dat'] = self.D_exact(dataset['x_dat'])


        dataset.subsample_evenly_astrain(dsopt['N_res_train'], ['x_res', 'D_res', 'f_res'])
        dataset.subsample_evenly_astrain(dsopt['N_dat_train'], ['x_dat', 'u_dat', 'D_dat'])

        self.dataset = dataset
    
    def create_dataset_from_file(self, dsopt):
        '''create dataset from file'''
        assert self.dataset is not None, 'datafile provide, dataset should not be None'
        uname = f'u{self.testcase}'
        dname = f'd{self.testcase}'

    
        self.dataset['x_dat'] = self.dataset['x']
        self.dataset['u_dat'] = self.dataset[uname]
        self.dataset['D_dat'] = self.dataset[dname]

        self.dataset['x_res'] = self.dataset['x']
        self.dataset['f_res'] = self.dataset['f']
        self.dataset['D_res'] = self.dataset[dname]
        
        self.dataset.subsample_evenly_astrain(dsopt['N_res_train'], ['x_res', 'D_res', 'f_res'])
        self.dataset.subsample_evenly_astrain(dsopt['N_dat_train'], ['x_dat', 'u_dat', 'D_dat'])

    def setup_dataset(self, dsopt, noise_opt):
        '''add noise to dataset'''
        if self.dataset is None:
            self.create_dataset_from_pde(dsopt)
        else:
            self.create_dataset_from_file(dsopt)

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
            self.dataset['upred_res'] = net(self.dataset['x_res'])
            coef = net.func_param(self.dataset['x_res'])
            self.dataset['func_res'] = coef


            self.dataset['upred_dat'] = net(self.dataset['x_dat'])
            coef = net.func_param(self.dataset['x_dat'])
            self.dataset['func_dat'] = coef
        
        # make prediction with different parameters
        self.prediction_variation(net)

    def prediction_variation(self, net):
        # make prediction with different parameters
        x = self.dataset['x_dat']
        D0 = self.dataset['D_dat']
        # first variation, D+0.1
        funs = {}
        funs['shitfplus']= lambda x: D0 + 0.1 * torch.ones_like(x)
        funs['shiftminus'] = lambda x: D0 - 0.1 * torch.ones_like(x)
        funs['linplus'] = lambda x: D0 + 0.1 * x
        funs['cosfull'] = lambda x: D0 + 0.1 * torch.cos(2*torch.pi * x)
        funs['sinfull'] = lambda x: D0 + 0.1 * torch.sin(2*torch.pi * x)
        funs['coshalf'] = lambda x: D0 + 0.1 * torch.cos(torch.pi * x)
        funs['sinhalf'] = lambda x: D0 + 0.1 * torch.sin(torch.pi * x)


        for funkey, fun in funs.items():
            # replace parameter
            with torch.no_grad():
                z = fun(x)
                u = net.variation(x, z )
                
            key = f'uvar_{funkey}_dat'
            var = f'Dvar_{funkey}_dat'
            self.dataset[key] = u
            self.dataset[var] = z

    
    def plot_variation(self, savedir=None):
        # go through uvar and var
        def get_funkey(key):
            return key.split('_')[1]
            

        for ukey in self.dataset.keys():
            if ukey.startswith('uvar'):
                fig, ax = plt.subplots(2,1)

                funkey = get_funkey(ukey)
                Dkey = ukey.replace('uvar', 'Dvar')
                # plot u
                ax[0].plot(self.dataset['x_dat'], self.dataset['u_dat'], label='u')
                ax[0].plot(self.dataset['x_dat'], self.dataset[ukey], label=funkey)
                ax[0].legend(loc="best")
                # plot var
                
                ax[1].plot(self.dataset['x_dat'], self.dataset['D_dat'], label='D')
                ax[1].plot(self.dataset['x_dat'], self.dataset[Dkey], label=funkey)
                ax[1].legend(loc="best")

                if savedir is not None:
                    path = os.path.join(savedir, f'fig_var_{funkey}.png')
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    print(f'fig saved to {path}')

        



    def validate(self, nn):
        '''compute l2 error and linf error of inferred D(x)'''
        x  = self.dataset['x_dat']
        D = self.dataset['D_dat']
        with torch.no_grad():
            Dpred = nn.func_param(x)
            l2norm = torch.mean(torch.square(D - Dpred))
            linfnorm = torch.max(torch.abs(D - Dpred)) 
        
        return {'l2err': l2norm.item(), 'linferr': linfnorm.item()}

    def plot_upred(self, savedir=None):
        fig, ax = plt.subplots()
        ax.plot(self.dataset['x_dat'], self.dataset['u_dat'], label='Exact')
        ax.plot(self.dataset['x_dat'], self.dataset['upred_dat'], label='NN')
        ax.scatter(self.dataset['x_dat_train'], self.dataset['u_dat_train'], label='data')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    def plot_Dpred(self, savedir=None):
        ''' plot predicted d and exact d'''
        fig, ax = plt.subplots()
        ax.plot(self.dataset['x_dat'], self.dataset['D_dat'], label='Exact')
        ax.plot(self.dataset['x_dat'], self.dataset['func_dat'], label='NN')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_D_pred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')

    def visualize(self, savedir=None):
        '''visualize the problem'''
        self.plot_upred(savedir)
        self.plot_Dpred(savedir)
        self.plot_variation(savedir)


if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.opts['nn_opts']['with_func'] = True
    optobj.opts['pde_opts']['problem'] = 'poivar'
    optobj.parse_args(*sys.argv[1:])
    
    
    device = set_device('cuda')
    set_seed(0)
    
    print(optobj.opts)

    prob = PoiVarProblem(**optobj.opts['pde_opts'])
    prob.setup_dataset(optobj.opts['dataset_opts'],optobj.opts['noise_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])


    prob.make_prediction(pdenet)
    prob.visualize(savedir=optobj.opts['logger_opts']['save_dir'])


