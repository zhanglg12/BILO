#!/usr/bin/env python
import sys

import torch
import torch.nn as nn

from util import *
from config import *
from MlflowHelper import MlflowHelper
from Problems import *
from Options import *
# the pde and the neural net is combined in one class
# the PDE parameter is also part of the network
class DensePoisson(nn.Module):
    def __init__(self, depth, width, input_dim=1, output_dim=1, 
                output_transform=lambda x,u:u,
                use_resnet=False, with_param=False, params_dict=None, 
                useFourierFeatures=False):
        super().__init__()
        
        
        self.depth = depth
        self.width = width
        self.use_resnet = use_resnet
        self.with_param = with_param # if True, then the pde parameter is part of the network
        self.output_transform = output_transform # transform the output of the network, default is identity
        self.useFourierFeatures = useFourierFeatures

        # convert float to tensor of size (1,1)
        # need ParameterDict to make it registered, otherwise to(device) will not automatically move it to device
        tmp = {k: nn.Parameter(torch.tensor([[v]])) for k, v in params_dict.items()}
        self.params_dict = nn.ParameterDict(tmp)


        if self.useFourierFeatures:
            print('Using Fourier Features')
            self.fflayer = nn.Linear(input_dim, width)
            self.fflayer.requires_grad = False
            self.input_layer = nn.Linear(width, width)
        else:
            self.input_layer = nn.Linear(input_dim, width)

        # depth = input + hidden + output
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 2)])
        self.output_layer = nn.Linear(width, output_dim)
        
        if use_resnet:
            self.residual_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 2)])
        
        

        # for with_param version, pde parameter is not part of the network (but part of module)
        # for inverse problem, create embedding layer for pde parameter
        if self.with_param == True:
            # Create embedding layers for each parameter
            self.param_embeddings = nn.ModuleDict({
                name: nn.Linear(1, width, bias=False) for name, param in self.params_dict.items()
            })
            # set requires_grad to False
            for param in self.param_embeddings.parameters():
                param.requires_grad = False


        # separate parameters for the neural net and the PDE parameters
        # neural net parameter exclude parameter embedding and fourier feature embedding layer
        self.param_net = list(self.input_layer.parameters()) +\
                            [param for layer in self.hidden_layers for param in layer.parameters()] +\
                            list(self.output_layer.parameters())

        self.param_pde = [self.params_dict[name] for name in self.params_dict.keys()]

        self.param_all = self.param_net + self.param_pde

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    

    def embedding(self, x):
        '''
        No fourier feature embedding:
            then y = x
        if fourier feature embedding:
            then y = sin(2*pi* (Wx+b))
        
        if with_param, then Wy+b
        if inverse, then Wy+b + W'p+b' (pde parameter embedding)

        '''
        # fourier feature embedding
        if self.useFourierFeatures:
            x = torch.sin(2 * torch.pi * self.fflayer(x))
        else:
            x = self.input_layer(x)

        if self.with_param:
            for name, param in self.params_dict.items():
                param_embedding = self.param_embeddings[name](param)
                x += param_embedding

        return x
        
    def forward(self, x):
        

        X = self.embedding(x)
        Xtmp = torch.tanh(X)
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden_output = torch.tanh(hidden_layer(Xtmp))
            if self.use_resnet:
                hidden_output += self.residual_layers[i](Xtmp)  # ResNet connection
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
    net = DensePoisson(**(opts['nn_opts']))
    net.load_state_dict(torch.load(artifact_paths['net.pth']))
    print(f'net loaded from {artifact_paths["net.pth"]}')
    return net, opts


# simple test of the network
# creat a network, compute residual, compute loss, no training
if __name__ == "__main__":

    
    optobj = Options()
    optobj.parse_args(*sys.argv[1:])
    

    device = set_device('cuda')
    set_seed(0)
    
    # prob = PoissonProblem(p=1, init_param={'D':1.0}, exact_param={'D':1.0})
    prob = create_pde_problem(**optobj.opts['pde_opts'])

    optobj.opts['nn_opts']['input_dim'] = prob.input_dim
    optobj.opts['nn_opts']['output_dim'] = prob.output_dim

    net = DensePoisson(**optobj.opts['nn_opts'],
                output_transform=prob.output_transform, 
                params_dict=prob.init_param).to(device)
    
    dataset = {}
    x = torch.linspace(0, 1, 20).view(-1, 1).to(device)
    x.requires_grad_(True)
    y = prob.u_exact(x, prob.exact_param)
    res, u_pred = prob.residual(net, x, net.params_dict)

    # print 2 norm of res
    print(torch.norm(res))
    

    