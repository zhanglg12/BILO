#!/usr/bin/env python
# used for study the scaling behavior of the neural network
import torch
from torchinfo import summary
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from BaseProblem import BaseProblem
from DataSet import DataSet

class TestProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = kwargs.get('input_dim', 1)
        self.output_dim = kwargs.get('output_dim', 1)
        self.tag=['exact']
        self.n = kwargs.get('n', 1)


        
        # dictionary of abcd .. input_dim
        for i in range(0,self.n+1):
            self.param['p'+str(i)] = 1.0
        
        self.opts = kwargs
        self.opts['init_param'] = self.param.copy()

        self.output_transform = lambda x, u: u

        self.idv = None
        self.msample = kwargs.get('msample', 0)
        self.net = None


    def residual(self, nn, x, param:dict):
        # synthetic residual, p0*u + p1*du/dx + p2*d^2u/dx^2 + ...
        x.requires_grad_(True)

        u_pred = nn(x)

        # compute n-th order derivative
        dus = 0
        
        dus = [u_pred] # list of u, du/dx, d^2u/dx^2, ...
        res = u_pred * param['p0']
        for i in range(0, self.n):
            du = torch.autograd.grad(dus[i], x,
                create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
            dus.append(du)
            res += param['p'+str(i+1)] * dus[i+1]
        
        self.res = res

        return res
    
    def setup_dataset(self, dsopt, noise_opt):
        '''add noise to dataset'''
        pass

    # copy from lossCollection
    def resgradloss(self):
        # compute gradient of residual w.r.t. parameter on a random sample of residual points
        self.res_unbind = self.res.unbind(dim=1) # unbind residual into a list of 1d tensor
        n = self.res.shape[0]

        # if msmple is 0, use all residual points
        # else, use JL
        if self.msample == 0:
            self.idv = torch.eye(n).to(self.res.device)    
        else:
            m = self.msample
            assert m <= n, 'msample should be less than the number of residual'
            n = self.res.shape[0]
            m = self.msample
            self.idv = torch.randn(m, n)/torch.sqrt(torch.tensor(m))
            self.idv = self.idv.to(self.res.device)
        
        
        resgradmse = torch.tensor(0.0, device=self.res.device) 
        for pname in self.net.trainable_param:
            for j in range(self.output_dim):
                tmp = torch.autograd.grad(self.res_unbind[j], self.net.params_dict[pname], grad_outputs=self.idv,
                create_graph=True, is_grads_batched=True, retain_graph=True)[0]
                resgradmse += torch.sum(torch.pow(tmp, 2))
        return resgradmse/n

if __name__ == "__main__":
    
    from Options import Options
    from util import set_device, set_seed, get_mem_stats
    from DenseNet import DenseNet
    import sys

    optobj = Options()
    optobj.parse_args(*sys.argv[2:])
    
    # first argument is the order
    n = sys.argv[1] 

    device = set_device('cuda')
    set_seed(0)
    
    prob = TestProblem(n=int(n), msample=optobj.opts['loss_opts']['msample'], trainable_param=optobj.opts['pde_opts']['trainable_param'])

    prob.net = prob.setup_network(**optobj.opts['nn_opts'])
    prob.net.to(device)

    summary(prob.net)
    
    for i in range(10):
        x = torch.rand(optobj.opts['dataset_opts']['N_res_train'], prob.input_dim, device=device)
        res = prob.residual(prob.net, x, prob.param)
        resnorm = torch.norm(res, p=2)
        print('res norm', resnorm.item())

        
        resgradmse = prob.resgradloss()
        print('resgrad:', resgradmse.item())
        
        # take grad w.r.t network parameters
        grad = torch.autograd.grad(resnorm + resgradmse, prob.net.param_net, create_graph=True,allow_unused=True)[0]

    mem = get_mem_stats()
    print(mem)
    