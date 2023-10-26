import torch
import torch.nn as nn
from util import *

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
            print(f'Epoch {epoch}, PDE Loss: {val_loss_res.item():.3g}, Data Loss: {val_loss_data.item():.3g} Total Loss: {val_loss_total.item():.3g} D loss: {val_loss_D.item():.3g} D: {net.D.item():.3g}')

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




        