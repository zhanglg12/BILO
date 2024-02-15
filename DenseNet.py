#!/usr/bin/env python
import sys

import torch
import torch.nn as nn

from util import *
from config import *
from MlflowHelper import MlflowHelper

from Options import *
# the pde and the neural net is combined in one class
# the PDE parameter is also part of the network
class DenseNet(nn.Module):
    def __init__(self, depth, width, input_dim=1, output_dim=1, 
                lambda_transform=lambda x, u: u,
                use_resnet=False, with_param=False, params_dict=None, 
                fourier=False,
                siren=False,
                with_func=False,
                trainable_param=[]):
        super().__init__()
        
        
        self.depth = depth
        self.width = width
        self.use_resnet = use_resnet
        self.with_param = with_param # if True, then the pde parameter is part of the network
        self.with_func = with_func # if True, then the unkonwn is a function
        self.lambda_transform = lambda_transform # transform the output of the network, default is identity
        self.fourier = fourier
        

        # convert float to tensor of size (1,1)
        # need ParameterDict to make it registered, otherwise to(device) will not automatically move it to device
        tmp = {k: nn.Parameter(torch.tensor([[v]])) for k, v in params_dict.items()}
        self.params_dict = nn.ParameterDict(tmp)
        self.params_expand = {}


        if self.fourier:
            print('Using Fourier Features')
            self.fflayer = nn.Linear(input_dim, width)
            self.fflayer.requires_grad = False
            self.input_layer = nn.Linear(width, width)
        else:
            self.input_layer = nn.Linear(input_dim, width)

        # depth = input + hidden + output
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 2)])
        self.output_layer = nn.Linear(width, output_dim)
        

        # for with_param version, pde parameter is not part of the network (but part of module)
        # for inverse problem, create embedding layer for pde parameter
        if self.with_param:
            # Create embedding layers for each parameter
            self.param_embeddings = nn.ModuleDict({
                name: nn.Linear(1, width, bias=True) for name, param in self.params_dict.items()
            })
            # set requires_grad to False
            for embedding_weights in self.param_embeddings.parameters():
                embedding_weights.requires_grad = False

        # for now, just one function
        if self.with_func:
            self.func_param = ParamFunction(depth=4, width=32)
            

        # activation function
        if siren:
            self.act = torch.sin
            self.siren_init()
        else:
            self.act = torch.tanh

        ### setup trainable parameters

        # trainable_param is a list of parameter names
        # only train part of the parameters

        # For now, set all PDE parameters to be trainable, for initialization, set lr=0
        # self.trainable_param = list(self.params_dict.keys())

        self.trainable_param = trainable_param
        for name, param in self.params_dict.items():
            if name not in self.trainable_param:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        self.collect_trainable_param()

        
       
    def collect_trainable_param(self):
        '''setup trianable parameter'''
        # separate parameters for the neural net and the PDE parameters
        # neural net parameter exclude parameter embedding and fourier feature embedding layer
        self.param_net = list(self.input_layer.parameters()) +\
                            [param for layer in self.hidden_layers for param in layer.parameters()] +\
                            list(self.output_layer.parameters())

        self.param_pde = list(self.params_dict.values())
        
        # For vanilla version, optimizer include all parameters
        # include untrainable parameters, so that the optimizer have the parameter in state_dict
        if not self.with_func:
            self.param_all = self.param_net + self.param_pde
        else:
            self.param_all = self.param_net + list(self.func_param.parameters())

        # collection of trainable parameters
        if not self.with_func:
            self.param_pde_trainable = [param for param in self.param_pde if param.requires_grad]
        else:
            self.param_pde_trainable = [param for param in self.func_param.parameters() if param.requires_grad]

    def output_transform(self, x, u):
        '''
        transform the output of the network
        '''
        return self.lambda_transform(x, u)
    
    def siren_init(self):
        '''
        initialize weights for siren
        '''
        self.omega_0 = 30
        with torch.no_grad():
            self.input_layer.weight.uniform_(-1 / self.input_layer.in_features, 1 / self.input_layer.in_features)
            for layer in self.hidden_layers:
                layer.weight.uniform_(-np.sqrt(6 / layer.in_features) / self.omega_0, 
                                             np.sqrt(6 / layer.in_features) / self.omega_0)
        

    def embedding(self, x, params_dict=None):
        '''
        No fourier feature embedding:
            then y = Wx+b = input_layer(x)
        if fourier feature embedding:
            then z = sin(2*pi* (Wx+b))
            then y = Wz+b = input_layer(z)
        
        if with_param, then Wy+b + W'p+b' (pde parameter embedding)
        This is the same as concat [x, pde_param] and then do a linear layer
        otherwise, then Wy+b

        '''
        
        # fourier feature embedding
        xcoord  = x
        if self.fourier:
            x = torch.sin(2 * torch.pi * self.fflayer(x))
        x = self.input_layer(x)

        # have to evaluate self.func_param(xcoord) inside the network
        # otherwise self.func_param is not in the computation graph
        if self.with_func is False:
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
        
    def forward(self, x, params_dict=None):
        
        X = self.embedding(x, params_dict)
        Xtmp = self.act(X)
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden_output = hidden_layer(Xtmp)
            if self.use_resnet:
                hidden_output += Xtmp  # ResNet connection
            hidden_output = self.act(hidden_output)
            Xtmp = hidden_output
        
        u = self.output_layer(Xtmp)
        u = self.output_transform(x, u)
        return u
    
    def reset_weights(self):
        '''
        Resetting model weights
        '''
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def freeze_layers_except(self, n):
        '''
        Freeze model, only the last n layers are trainable
        '''
        self.param_net = []
        total = len(list(self.children()))
        print(f'total {total} layers, freeze {total-n} layers')
        for i, layer in enumerate(self.children()):
            if i < total - n:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
                self.param_net += list(layer.parameters())
    
    def print_named_module(self):
        for name, layer in self.named_modules():
            print(name, layer)



def load_artifact(exp_name=None, run_name=None, run_id=None, name_str=None):
    """ 
    Load options and artifact paths from mlflow run id or name
    """
    if name_str is not None:
        try:
            exp_name, run_name = name_str.split(':')
        except ValueError:
            raise ValueError("name_str must be in the format 'exp_name:run_name'")

    helper = MlflowHelper()        
    if run_id is None:
        run_id = helper.get_id_by_name(exp_name, run_name)

    artifact_paths = helper.get_active_artifact_paths(run_id)
    opts = read_json(artifact_paths['options.json'])
    return opts, artifact_paths


def load_model(exp_name=None, run_name=None, run_id=None, name_str=None):
    """ 
    easy load model from mlflow run id or name
    """
    opts, artifact_paths = load_artifact(exp_name, run_name, run_id, name_str)
    
    # reconstruct net from options and load weight
    net = DenseNet(**(opts['nn_opts']))
    net.load_state_dict(torch.load(artifact_paths['net.pth']))
    print(f'net loaded from {artifact_paths["net.pth"]}')
    return net, opts


class ParamFunction(nn.Module):
    '''represent unknown f(x) to be learned, diffusion field or initial condition'''
    def __init__(self, depth=4, width=16, 
                 activation='tanh', output_activation='softplus', 
                 output_transform=lambda x, u: u):
        super(ParamFunction, self).__init__()
        
        # represent single variable function
        input_dim = 1
        output_dim = 1

        if activation == 'tanh':
            activation = nn.Tanh
        elif activation == 'relu':
            activation = nn.ReLU
        elif activation == 'sigmoid':
            activation = nn.Sigmoid
        elif activation == 'id':
            activation = nn.Identity
        else:
            raise ValueError('activation function not supported')

        if output_activation == 'softplus':
            output_activation = nn.Softplus
        elif output_activation == 'id':
            output_activation = nn.Identity
        elif output_activation == 'relu':
            output_activation = nn.ReLU
        else:
            raise ValueError('output activation function not supported')


        # Create the layers of the neural network
        layers = []
        if depth == 1:
            # Only one layer followed by output_activation if depth is 1
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(output_activation())
        else:
            # input layer
            layers.append(nn.Linear(input_dim, width))
            layers.append(activation())

            # hidden layers (depth - 2 because we already have input and output layers)
            for _ in range(depth - 2):
                layers.append(nn.Linear(width, width))
                layers.append(activation())

            # output layer
            layers.append(nn.Linear(width, output_dim))
            layers.append(output_activation())

        # Store the layers as a sequential module
        self.layers = nn.Sequential(*layers)

        # Store the output transformation function
        self.output_transform = output_transform

    def forward(self, x):
        # Define the forward pass
        u = self.layers(x)
        return self.output_transform(x, u)

# simple test of the network
# creat a network, compute residual, compute loss, no training
# if __name__ == "__main__":

    
#     optobj = Options()
#     optobj.parse_args(*sys.argv[1:])
    

#     device = set_device('cuda')
#     set_seed(0)
    
#     prob = create_pde_problem(**optobj.opts['pde_opts'])
#     prob.print_info()

#     optobj.opts['nn_opts']['input_dim'] = prob.input_dim
#     optobj.opts['nn_opts']['output_dim'] = prob.output_dim

#     net = DenseNet(**optobj.opts['nn_opts'],
#                 output_transform=prob.output_transform, 
#                 params_dict=prob.param).to(device)
    
#     dataset = {}
#     x = torch.linspace(0, 1, 20).view(-1, 1).to(device)
#     x.requires_grad_(True)
#     y = prob.u_exact(x, prob.param)
#     res, u_pred = prob.residual(net, x, net.params_dict)

#     jac  = prob.compute_jacobian(net, x, net.params_dict)


#     # print 2 norm of res
#     print('res = ',torch.norm(res))
#     print('jac = ',torch.norm(jac)) 
    

    