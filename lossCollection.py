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

from DataSet import DataSet
from util import *

class lossCollection:
    # loss, parameter, and optimizer
    def __init__(self, net, pde, dataset, loss_weigt_dict):

        
        self.net = net
        self.pde = pde
        self.dataset = dataset
        


        # intermediate results for residual loss
        self.res = None
        self.mse_res = None
        self.grad_res_params = {} # gradient of residual w.r.t. parameter
        self.u_pred = None

        # collection of all loss functions
        self.loss_dict = {'res': self.resloss, 'resgrad': self.resgradloss, 'data': self.dataloss, 'paramgrad': self.paramgradloss}
        
        self.loss_weight = loss_weigt_dict # dict of active loss: weight
        
        # collect keys with positive weights
        self.loss_active = []
        for k in self.loss_weight.keys():
            if self.loss_weight[k] is not None:
                self.loss_active.append(k)

        # if residual gradient loss is active, we need to compute residual gradient
        if 'res' in self.loss_active:
            self.dataset['x_res_train'].requires_grad_(True)

    
        self.wloss_comp = {} # component of each loss, weighted
        self.wtotal = None # total loss for backprop


    def computeResidual(self):
        self.dataset['x_res_train'].requires_grad_(True)
        self.res, self.u_pred = self.pde.residual(self.net, self.dataset['x_res_train'], self.net.params_dict)
        return self.res, self.u_pred

    # def computeResidualGrad(self):
    #     # compute gradient of residual w.r.t. parameter
    #     for pname in self.net.trainable_param:
    #         # this does not seem right, lots of cancelation
    #         self.grad_res_params[pname] = torch.autograd.grad(self.res, self.net.params_dict[pname], create_graph=True, grad_outputs=torch.ones_like(self.res))[0]
    #     return self.grad_res_params

    # def computeResidualGrad(self):
    #     # super slow
    #     # compute gradient of residual w.r.t. parameter
    #     res1d = self.res.flatten()
    #     for pname in self.net.trainable_param:
    #         sum_grad = 0.0
    #         for i in range(res1d.numel()):
    #             grad = torch.autograd.grad(res1d[i], self.net.params_dict[pname], create_graph=True)[0]
    #             sum_grad += grad**2
    #         self.grad_res_params[pname] = sum_grad

    #     return self.grad_res_params
        

    def resloss(self):
        self.computeResidual()
        val_loss_res = mse(self.res)
        self.mse_res = val_loss_res
        return val_loss_res
    
    # def resgradloss(self):
    #     # compute mse of (gradient of residual w.r.t. parameter)
    #     self.computeResidualGrad()
        
    #     # compute the sum of squares of grad-res w.r.t parameters
    #     values = torch.stack(list(self.grad_res_params.values()))
    #     mse = torch.mean(torch.pow(values, 2))
        
    #     return mse

    def resgradloss(self):
        # hacky, not exactly what I want
        # this is MSE of MSE-res w.r.t parameters
        # not MSE of res w.r.t parameters
        sum_grad_sqr = 0.0
        for pname in self.net.trainable_param:
            # this does not seem right, lots of cancelation
            self.grad_res_params[pname] = torch.autograd.grad(self.mse_res, self.net.params_dict[pname], create_graph=True)[0]
            sum_grad_sqr += (self.grad_res_params[pname])**2
        mse = sum_grad_sqr

        return mse
            
        
    # to prevent derivative of u w.r.t. parameter to be 0
    # for now, just fix the weight of the embedding.
    def paramgradloss(self):
        pass
    #     # derivative of u w.r.t. D
    #     # penalty term to keep away from 0
    #     return torch.exp(-mse(self.u_D))
    
    def dataloss(self):
        # a little bit less efficient, u_pred is already computed in resloss
        self.u_pred = self.net(self.dataset['x_dat_train'])
        return mse(self.u_pred, self.dataset['u_dat_train'])
    
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

    def __call__(self, loss, params, epoch):
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
    from DensePoisson import *
    from Problems import *


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

    dataset = create_dataset_from_pde(prob, optobj.opts['dataset_opts'], optobj.opts['noise_opts'])
    dataset.to_device(device)

    dataset['u_dat_train'] = dataset['u_exact_dat_train']

    params = list(net.parameters())
    


