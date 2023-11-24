import torch
import torch.nn as nn
import torch.optim as optim
from util import *
from config import *
from MlflowHelper import MlflowHelper

# the pde and the neural net is combined in one class
# the PDE parameter is also part of the network
class DensePoisson(nn.Module):
    def __init__(self, depth, width, use_resnet=False, basic=False, init_D=1.0, p = 1, useFourierFeatures=False):
        super().__init__()
        
        
        self.depth = depth
        self.width = width
        self.use_resnet = use_resnet
        self.init_D = init_D
        self.basic = basic
        self.useFourierFeatures = useFourierFeatures
        
        self.p = p  # determine class of problem, larger p -> more oscillations

        if self.useFourierFeatures:
            print('Using Fourier Features')
            self.fflayer = nn.Linear(1, width)
            self.fflayer.requires_grad = False
            self.input_layer = nn.Linear(width, width)
        else:
            self.input_layer = nn.Linear(1, width)

        # depth = input + hidden + output
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 2)])
        self.output_layer = nn.Linear(width, 1)
        
        if use_resnet:
            self.residual_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 2)])
        
        self.D = nn.Parameter(torch.tensor([init_D]))  # initialize D to 1

        # for basic version, D is not part of the network
        if self.basic == False:
            self.fcD = nn.Linear(1, width, bias=False)
            self.fcD.requires_grad = False


        # separate parameters for the neural net and the PDE parameter
        # self.param_net = [param for param in self.parameters() if (param is not self.D) and (param is not self.fcD)]

        self.param_net = list(self.input_layer.parameters()) +\
                            [param for layer in self.hidden_layers for param in layer.parameters()] +\
                            list(self.output_layer.parameters())

        self.param_pde = [self.D]
        

    def forward(self, x):
        
        # fbc = torch.sin(torch.pi * x) # transformation of nn to impose boundary condition
        fbc = x * (1 - x) 


        if self.useFourierFeatures:
            x = torch.sin(2 * torch.pi * self.fflayer(x))
   
        if self.basic == False:
            x = torch.sigmoid(self.input_layer(x) + self.fcD(self.D))
        else:
            x = torch.sigmoid(self.input_layer(x))
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden_output = torch.sigmoid(hidden_layer(x))
            if self.use_resnet:
                hidden_output += self.residual_layers[i](x)  # ResNet connection
            x = hidden_output
        
        u = (self.output_layer(x)) * fbc
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

    artifact_paths = helper.get_artifact_paths(run_id)
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


# set up simple test
if __name__ == "__main__":
    device = set_device('cpu')

    # basic version
    # net = DensePoisson(2,6,basic=True).to(device)
    
    net = DensePoisson(2,6).to(device)

    exact_D = 2.0
    dataset = {}
    dataset['x_res_train'] = torch.linspace(0, 1, 20).view(-1, 1).to(device)
    dataset['x_res_train'].requires_grad_(True)
    dataset['u_res_train'] = net.u_exact(dataset['x_res_train'], exact_D)

    lossopt = {'weights':{'res':1.0,'data':1.0}}

    # train basic version
    # lossObj = lossCollection(net, dataset, list(net.parameters()), optim.Adam, lossopt)

    # train inverse problem
    loss_pde_opts = {'weights':{'res':1.0,'resgrad':1.0}}
    loss_pde = lossCollection(net, dataset, net.param_net, optim.Adam, loss_pde_opts)

    loss_data_opts = {'weights':{'data':1.0}}
    loss_data = lossCollection(net, dataset, net.param_pde, optim.Adam, loss_data_opts)
