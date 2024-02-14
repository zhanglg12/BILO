#!/usr/bin/env python
'''
this class handle the loss function and computing gradient
net = neural net class
dataset = data set class,
pde = pde class, take net and dataset to compute residual
param = list of parameters to optimize, either network weight or pde parameter
lossCollection compute different loss, in particular 
residual loss: residual of pde 
residual gradient loss: derivative of residual w.r.t. pde parameter
data loss: MSE of data
'''
import torch
from util import mse, set_device, set_seed

class lossCollection:
    # loss, parameter, and optimizer
    def __init__(self, net, pde,  loss_weight_dict, loss_opts):


        self.net = net
        self.pde = pde

        # intermediate results for residual loss
        self.res = None
        self.grad_res_params = {} # gradient of residual w.r.t. parameter
        self.u_pred = None

        # collection of all loss functions
        self.loss_dict = {'res': self.resloss,
        'fullresgrad': self.fullresgradloss, 'data': self.dataloss, 'paramgrad': self.paramgradloss,
        'bc': self.bcloss,'funcloss':self.funcloss,
        'resgradfunc': self.resgradfuncloss}

        self.loss_weight = {} # dict of active loss: weight

        # collect keys with positive weights
        self.loss_active = []
        for k in self.loss_dict.keys():
            if loss_weight_dict[k] is not None:
                self.loss_active.append(k)
                self.loss_weight[k] = loss_weight_dict[k]
    
        self.wloss_comp = {} # component of each loss, weighted
        self.wtotal = None # total loss for backprop

        self.idmx = None # identity matrix for computing gradient of residual w.r.t. parameter
        self.idv = None # sampling matrix
        self.msample = loss_opts['msample']
    
    def resloss(self):
        self.res, self.u_pred = self.pde.get_res_pred(self.net)
        val_loss_res = mse(self.res)
        return val_loss_res

    def resgradfuncloss(self):
        # compute gradient of residual w.r.t. parameter on every residual point.
        # very memory intensive
        # assume output dim is 1

        n = self.res.shape[0]
        if self.idmx is None:
            self.idmx = torch.eye(n).to(self.res.device) # identity matrix for computing gradient of residual w.r.t. parameter

        resgradmse = 0.0
        res_flat = torch.flatten(self.res)
        
        tmp = torch.autograd.grad(res_flat, self.net.param_pde_trainable, grad_outputs=self.idmx,
            create_graph=True, retain_graph=True,allow_unused=True, is_grads_batched=True)

        # mse
        for p in tmp:
            resgradmse += torch.sum(torch.pow(p, 2))
        
        return resgradmse/n

    def fullresgradloss(self):
        # compute gradient of residual w.r.t. parameter on every residual point.
        # very memory intensive
        self.res_unbind = self.res.unbind(dim=1) # unbind residual into a list of 1d tensor

        n = self.res.shape[0]
        if self.idmx is None:
            self.idmx = torch.eye(n).to(self.res.device) # identity matrix for computing gradient of residual w.r.t. parameter

        resgradmse = 0.0        
        for pname in self.net.trainable_param:
            for j in range(self.pde.output_dim):
                tmp = torch.autograd.grad(self.res_unbind[j], self.net.params_expand[pname], grad_outputs=torch.ones_like(self.res_unbind[j]),
                create_graph=True, retain_graph=True,allow_unused=True)[0]

                resgradmse += torch.sum(torch.pow(tmp, 2))
        
        return resgradmse/n

    # to prevent derivative of u w.r.t. parameter to be 0
    # for now, just fix the weight of the embedding.
    def paramgradloss(self):
        pass
    #     # derivative of u w.r.t. D
    #     # penalty term to keep away from 0
    #     return torch.exp(-mse(self.u_D))
    
    def dataloss(self):
        # a little bit less efficient, u_pred is already computed in resloss
        return self.pde.get_data_loss(self.net)
    
    def bcloss(self):
        # compute loss from boundary condition
        return self.pde.get_bc_loss(self.net)
    
    def funcloss(self):
        # compute loss from parameter
        return self.pde.func_mse(self.net)
    
    def getloss(self):
        # for each active loss, compute the loss and multiply with the weight
        losses = {}
        weighted_sum = 0
        for key in self.loss_active:
            losses[key] = self.loss_weight[key] * self.loss_dict[key]()
            weighted_sum += losses[key]
        
        self.wloss_comp = losses
        self.wtotal = weighted_sum
        self.wloss_comp['total'] = self.wtotal
    
    def get_wloss_sum(self, list_of_loss):
        # return the sum of weighted loss
        return sum([self.wloss_comp[key] for key in list_of_loss])


class EarlyStopping:
    def __init__(self,  **kwargs):
        self.tolerance = kwargs.get('tolerance', 1e-4)
        self.max_iter = kwargs.get('max_iter', 10000)
        self.patience = kwargs.get('patience', 100)
        self.delta_loss = kwargs.get('delta_loss', 0)
        self.burnin = kwargs.get('burnin',1000 )
        self.monitor_loss = kwargs.get('monitor_loss', True)
        self.best_loss = None
        self.counter_param = 0
        self.counter_loss = 0
        self.epoch = 0

    def __call__(self, loss, params, epoch):
        self.epoch = epoch
        if epoch >= self.max_iter:
            print('\nStop due to max iteration')
            return True
        
        if loss < self.tolerance:
            print('Stop due to loss {loss} < {self.tolerance}')
            return True
         
        if epoch < self.burnin:
            return False

        if self.monitor_loss:
            if self.best_loss is None:
                self.best_loss = loss
            elif loss > self.best_loss - self.delta_loss:
                self.counter_loss += 1
                if self.counter_loss >= self.patience:
                    print(f'Stop due to loss patience for {self.counter_loss} steps, best loss {self.best_loss}')
                    return True
            else:
                self.best_loss = loss
                self.counter_loss = 0
        return False


if __name__ == "__main__":


    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.parse_args(*sys.argv[1:])
    
    device = set_device('cuda')
    set_seed(0)
    
    # prob = PoissonProblem(p=1, init_param={'D':1.0}, exact_param={'D':1.0})
    prob = create_pde_problem(**optobj.opts['pde_opts'])

    optobj.opts['nn_opts']['input_dim'] = prob.input_dim
    optobj.opts['nn_opts']['output_dim'] = prob.output_dim

    net = DenseNet(**optobj.opts['nn_opts'],
                output_transform=prob.output_transform, 
                params_dict=prob.init_param).to(device)

    dataset = create_dataset_from_pde(prob, optobj.opts['dataset_opts'], optobj.opts['noise_opts'])
    dataset.to_device(device)

    dataset['u_dat_train'] = dataset['u_exact_dat_train']

    params = list(net.parameters())
    


