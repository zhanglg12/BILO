import torch
import torch.nn as nn
import torch.optim as optim
from util import *
from config import *




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

        self.param_net = all_params = list(self.input_layer.parameters()) +\
                            [param for layer in self.hidden_layers for param in layer.parameters()] +\
                            list(self.output_layer.parameters())

        self.param_pde = [self.D]
        

    def forward(self, x):
        
        # fbc = torch.sin(torch.pi * x) # transformation of nn to impose boundary condition
        fbc = x * (1 - x) 


        if self.useFourierFeatures:
            x = torch.sin(2 * torch.pi * self.fflayer(x))

        x = torch.sigmoid(self.input_layer(x))
   
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
    
    
    # define residual
    def residual(self, x, D):
        u_pred = self.forward(x)
        
        u_x = torch.autograd.grad(u_pred, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True,retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        u_D = torch.autograd.grad(u_pred, self.D, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]

        res = D * u_xx - self.f(x)

        # differntiate res w.r.t. D
        res_D = torch.autograd.grad(res, self.D, create_graph=True, grad_outputs=torch.ones_like(res))[0]
        return res, res_D, u_pred, u_D
    
    def f(self, x):
        return - (torch.pi * self.p)**2 * torch.sin(torch.pi * self.p * x)
    
    def u_exact(self, x, D):
        return torch.sin(torch.pi * self.p * x) / D

    def u_init(self, x):
        return torch.sin(torch.pi * self.p * x) / self.init_D



class lossCollection:
    # loss, parameter, and optimizer
    def __init__(self, net, dataset, param, optimizer, opts):

        self.opts = opts
        self.net = net
        self.dataset = dataset
        self.optimizer = optimizer(param)
        self.param = param

        # intermediate results for residual loss
        self.res = None
        self.res_D = None
        self.u_pred = None
        self.u_D = None

        # collection of all loss functions
        self.loss_dict = {'res': self.resloss, 'resgrad': self.resgradloss, 'data': self.dataloss, 'paramgrad': self.paramgradloss}
        
        self.loss_weight = opts['weights']
        
        # collect keys with positive weights
        self.loss_active = []
        for k in self.loss_weight.keys():
            if self.loss_weight[k] is not None and self.loss_weight[k] > 0:
                self.loss_active.append(k)

    
        self.loss_comp = {} # component of each loss
        self.loss_val = None # total loss for backprop


    def computeResidual(self):
        self.res, self.res_D, self.u_pred, self.u_D = self.net.residual(self.dataset['x_res_train'], self.net.D)
    


    def resloss(self):
        self.computeResidual()
        val_loss_res = mse(self.res)
        return val_loss_res
    
    def resgradloss(self):
        return mse(self.res_D)
    
    def paramgradloss(self):
        return torch.exp(-mse(self.u_D))
    
    def dataloss(self):
        # a little bit less efficient, u_pred is already computed in resloss
        self.u_pred = self.net(self.dataset['x_res_train'])
        return mse(self.u_pred, self.dataset['u_res_train'])
    
    def getloss(self):
        # for each active loss, compute the loss and multiply with the weight
        losses = {}
        weighted_sum = 0
        for key in self.loss_active:
            losses[key] = self.loss_weight[key] * self.loss_dict[key]()
            weighted_sum += losses[key]
        
        self.loss_comp = losses
        self.loss_val = weighted_sum

    def step(self):
        self.optimizer.zero_grad()
        # 1 step of gradient descent
        grads = torch.autograd.grad(self.loss_val, self.param, create_graph=True, allow_unused=True)
        for param, grad in zip(self.param, grads):
            param.grad = grad
        self.optimizer.step()



# set up simple test
if __name__ == "__main__":
    device = set_device('cpu')

    # basic version
    # net = DensePoisson(2,6,basic=True).to(device)
    
    net = DensePoisson(2,6).to(device)

    Dexact = 2.0
    dataset = {}
    dataset['x_res_train'] = torch.linspace(0, 1, 20).view(-1, 1).to(device)
    dataset['x_res_train'].requires_grad_(True)
    dataset['u_res_train'] = net.u_exact(dataset['x_res_train'], Dexact)

    lossopt = {'weights':{'res':1.0,'data':1.0}}

    # train basic version
    # lossObj = lossCollection(net, dataset, list(net.parameters()), optim.Adam, lossopt)

    # train inverse problem
    loss_pde_opts = {'weights':{'res':1.0,'resgrad':1.0}}
    loss_pde = lossCollection(net, dataset, net.param_net, optim.Adam, loss_pde_opts)

    loss_data_opts = {'weights':{'data':1.0}}
    loss_data = lossCollection(net, dataset, net.param_pde, optim.Adam, loss_data_opts)
    
    trainopt = {'max_iter': 1000, 'print_every':10, 'tolerance':1e-2}
    train_network(net, [loss_pde, loss_data], trainopt)