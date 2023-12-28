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
    def __init__(self, net, pde, dataset, param, optimizer, opts):

        self.opts = opts
        self.net = net
        self.pde = pde
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

    
        self.wloss_comp = {} # component of each loss, weighted
        self.loss_val = None # total loss for backprop


    def computeResidual(self):
        
        self.res, self.u_pred = self.pde.residual(self.net, self.dataset['x_res_train'], self.net.params_dict)
        return self.res, self.u_pred

    def computeResidualGrad(self):
        # compute gradient of residual w.r.t. parameter
        self.res_D = torch.autograd.grad(self.res, self.net.params_dict['D'], create_graph=True, grad_outputs=torch.ones_like(self.res))[0]
        return self.res_D
        

    def resloss(self):
        self.computeResidual()
        val_loss_res = mse(self.res)
        return val_loss_res
    
    def resgradloss(self):
        # compute gradient of residual w.r.t. parameter
        self.computeResidualGrad()
        return mse(self.res_D)
    
    def paramgradloss(self):
        # derivative of u w.r.t. D
        # penalty term to keep away from 0
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
        
        self.wloss_comp = losses
        self.loss_val = weighted_sum

    def step(self):
        self.optimizer.zero_grad()
        # 1 step of gradient descent
        grads = torch.autograd.grad(self.loss_val, self.param, create_graph=True, allow_unused=True)
        for param, grad in zip(self.param, grads):
            param.grad = grad
        self.optimizer.step()
    
    def save_state(self, fpath):
        # save optimizer state
        torch.save(self.optimizer.state_dict(), fpath)


def setup_dataset(pde, noise_opts, ds_opts):
    # set up data set according to PDE, traintype, and noise_opts, ds_opts
    
    dataset = DataSet()
    xtmp = torch.linspace(0, 1, ds_opts['N_res_train'] ).view(-1, 1)
    
    dataset['x_res_train'] = xtmp
    # dataset['x_res_train'].requires_grad_(True)

    dataset['x_res_test'] = torch.linspace(0, 1, ds_opts['N_res_test']).view(-1, 1)

    # generate data, might be noisy
    dataset['u_res_train'] = pde.u_exact(dataset['x_res_train'], pde.exact_D)


    if noise_opts and noise_opts['use_noise']:
        dataset['noise'] = generate_grf(xtmp, noise_opts['variance'],noise_opts['length_scale'])
        dataset['u_res_train'] = dataset['u_res_train'] + dataset['noise']
        print('Noise added')
    else:
        print('No noise added')
    
    return dataset


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
            print('Stop due to max iteration')
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


# if __name__ == "__main__":
#     # test to check if the loss can be computed
