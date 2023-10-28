import torch
import torch.nn as nn
import torch.optim as optim
from util import *
from config import *

# the pde and the neural net is combined in one class
# the PDE parameter is also part of the network

class DensePoisson(nn.Module):
    def __init__(self, depth, width, use_resnet=False, basic=False, init_D=1.0, p = 1):
        super().__init__()
        
        self.depth = depth
        self.width = width
        self.use_resnet = use_resnet
        self.init_D = init_D
        self.basic = basic
        
        self.p = p  # determine class of functions, larger p -> more oscillations

        self.input_layer = nn.Linear(1, width)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 2)])
        self.output_layer = nn.Linear(width, 1)
        
        if use_resnet:
            self.residual_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 2)])
        
        self.D = nn.Parameter(torch.tensor([init_D]))  # initialize D to 1

        # for basic version, D is not part of the network
        if self.basic == False:
            self.fcD = nn.Linear(1, width)


        # separate parameters for the neural net and the PDE parameter
        self.param_net = [param for param in self.parameters() if param is not self.D]
        self.param_pde = [self.D]
        

    def forward(self, x):
        fbc = torch.sin(torch.pi * x)
        if self.basic == False:
            x = torch.sigmoid(self.input_layer(x) + self.fcD(self.D))
        else:
            x = torch.sigmoid(self.input_layer(x))
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden_output = torch.sigmoid(hidden_layer(x))
            if self.use_resnet:
                hidden_output += self.residual_layers[i](x)  # ResNet connection
            x = hidden_output
        
        u = self.output_layer(x) * fbc
        return u
    
    
    # define residual
    def residual(self, x, D):
        u_pred = self.forward(x)
        u_x = torch.autograd.grad(u_pred, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True,retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = D * u_xx - self.f(x)

        # differntiate res w.r.t. D
        res_D = torch.autograd.grad(res, self.D, create_graph=True, grad_outputs=torch.ones_like(res))[0]
        return res, res_D, u_pred
    
    def f(self, x):
        return - (torch.pi * self.p)**2 * torch.sin(torch.pi * self.p * x)
    
    def u_exact(self, x, D):
        return torch.sin(torch.pi * self.p * x) / D

    def u_init(self, x):
        return torch.sin(torch.pi * self.p * x) / self.init_D



def train_network_init(net, optimizer_net, dataset, opts):
    
    epoch = 0
    u_data_init = net.u_init(dataset['x_train_res']) # initial value of u to help training
    while True:
        # Zero the gradients
        optimizer_net.zero_grad()
        
        res, res_D, u_pred = net.residual(dataset['x_train_res'], net.D)
        # Forward pass
        val_loss_res = mse(res)
        val_loss_D = mse(res_D)

        val_loss_data = mse(u_pred, u_data_init )

        
        val_loss_total = val_loss_res + val_loss_data + val_loss_D
        
        # Backward pass
        grads_net = torch.autograd.grad(val_loss_total, net.param_net, create_graph=True, allow_unused=True)
        
        for param, grad in zip(net.param_net, grads_net):
            param.grad = grad
        
        # Step the optimizers
        optimizer_net.step()
        
        # Output loss values up to three significant digits
        if epoch % opts['print_every'] == 0:
            # print(f'Epoch {epoch}, PDE Loss: {val_loss_res.item():.3g}, Data Loss: {val_loss_data.item():.3g} Total Loss: {val_loss_total.item():.3g} D loss: {val_loss_D.item():.3g} D: {net.D.item():.3g}')
            print_statistics(epoch, PDE=val_loss_res.item())

        # Termination conditions
        if val_loss_total.item() < opts['tolerance'] or epoch >= opts['max_iter']:
            break  # Exit the loop if loss is below tolerance or maximum iterations reached
        
        epoch += 1  # Increment the epoch counter


def train_network_inverse(net, optimizer_net, optimizer_D, dataset, opts):
    
    # Training loop
    epoch = 0
    while True:  # Change to a while loop to allow for early termination
        # Zero the gradients
        optimizer_net.zero_grad()
        optimizer_D.zero_grad()
        
        # Forward pass
        res, res_D, u_pred = net.residual(dataset['x_train_res'], net.D)
        # Forward pass
        val_loss_res = mse(res)
        val_loss_data = mse(u_pred, dataset['u_data'])
        val_loss_D = mse(res_D)
        
        val_loss_total = val_loss_res + val_loss_data + val_loss_D

        # Backward pass
        grads_net = torch.autograd.grad(val_loss_res + val_loss_D, net.param_net, create_graph=True, allow_unused=True)
        grads_pdeparam = torch.autograd.grad(val_loss_data, net.param_pde, create_graph=True, allow_unused=True)
        
        for param, grad in zip(net.param_net, grads_net):
            param.grad = grad
        
        for param, grad in zip(net.param_pde, grads_pdeparam):
            param.grad = grad

        # Step the optimizers
        optimizer_net.step()
        optimizer_D.step()

        # Output loss values up to three significant digits
        if epoch % opts['print_every'] == 0:
            print_statistics(epoch, PDE=val_loss_res.item(), Data=val_loss_data.item(), Dloss=val_loss_D.item(), Total=val_loss_total.item(), D=net.D.item())

        # Termination conditions
        if val_loss_total.item() < opts['tolerance'] or epoch >= opts['max_iter']:
            break  # Exit the loop if loss is below tolerance or maximum iterations reached

        epoch += 1  # Increment the epoch counter


def train_network_vanilla(net, optimizer_full, dataset, opts):
    # vanilla version of inverse problem
    epoch = 0
    u_data_init = net.u_init(dataset['x_train_res']) # initial value of u to help training
    while True:
        # Zero the gradients
        optimizer_full.zero_grad()
        
        res, res_D, u_pred = net.residual(dataset['x_train_res'], net.D)
        # Forward pass
        val_loss_res = mse(res)

        val_loss_data = mse(u_pred, dataset['u_data'] )

        val_loss_total = val_loss_res + val_loss_data
        
        # Backward pass
        val_loss_total.backward(retain_graph=True)
        
        # Step the optimizers
        optimizer_full.step()
        
        # Output loss values up to three significant digits
        if epoch % opts['print_every'] == 0:
            print_statistics(epoch, PDE=val_loss_res.item(), Data=val_loss_data.item(), Total=val_loss_total.item(), D=net.D.item())

        # Termination conditions
        if val_loss_total.item() < opts['tolerance'] or epoch >= opts['max_iter']:
            break  # Exit the loop if loss is below tolerance or maximum iterations reached
        
        epoch += 1  # Increment the epoch counter


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

        # collection of all loss functions
        self.loss_dict = {'res': self.resloss, 'resgrad': self.resgradloss, 'data': self.dataloss}
        
        self.loss_weight = opts['weights']
        
        # collect keys with positive weights
        self.loss_active = []
        for k in self.loss_weight.keys():
            if self.loss_weight[k] is not None and self.loss_weight[k] > 0:
                self.loss_active.append(k)

    
        self.loss_comp = {} # component of each loss
        self.loss_val = None # total loss for backprop


    def computeResidual(self):
        self.res, self.res_D, self.u_pred = self.net.residual(self.dataset['x_train_res'], self.net.D)

    def resloss(self):
        self.computeResidual()
        val_loss_res = mse(self.res)
        return val_loss_res
    
    def resgradloss(self):
        return mse(self.res_D)
    
    def dataloss(self):
        # a little bit less efficient, u_pred is already computed in resloss
        self.u_pred = self.net(self.dataset['x_train_res'])
        return mse(self.u_pred, self.dataset['u_data'])
    
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
    dataset['x_train_res'] = torch.linspace(0, 1, 20).view(-1, 1).to(device)
    dataset['x_train_res'].requires_grad_(True)
    dataset['u_data'] = net.u_exact(dataset['x_train_res'], Dexact)

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