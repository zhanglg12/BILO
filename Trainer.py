#!/usr/bin/env python

# for training the network
# need: options, network, pde, dataset, lossCollection
import os
import time
import torch
import torch.optim as optim
from lossCollection import lossCollection, EarlyStopping
from Logger import Logger
from util import get_mem_stats, set_device, set_seed, print_dict, flatten

class Trainer:
    def __init__(self, opts, net, pde, device, lossCollection, logger:Logger):
        self.opts = opts
        self.logger = logger
        self.net = net
        self.pde = pde
        self.device = device
        self.info = {}

        
        if opts['whichoptim'] == 'adam':
            self.optim = optim.Adam
        elif opts['whichoptim'] == 'adamw':
            self.optim = optim.AdamW
        else:
            raise ValueError(f'optimizer {opts["whichoptim"]} not supported')

        self.lossCollection = lossCollection
        
        self.optimizer = {}

        # early stopping
        self.estop = EarlyStopping(**self.opts)

        self.loss_net = opts['loss_net']
        self.loss_pde = opts['loss_pde']

        self.info['num_params'] = sum(p.numel() for p in self.net.parameters())
        self.info['num_train_params'] = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        

    

    def log_stat(self, wloss_comp, epoch):
        # log statistics
        if wloss_comp is None:
            return
        
        self.logger.log_metrics(wloss_comp, step=epoch)
        if not self.net.with_func:
            # log network parameters if param not function
            self.logger.log_metrics(self.net.params_dict, step=epoch)
    
    def set_grad(self, params, loss):
        '''
        set gradient of loss w.r.t params
        '''
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        for param, grad in zip(params, grads):
            param.grad = grad
    
    def set_pde_param_grad(self, loss):
        '''
        set gradient of loss w.r.t pde parameter
        '''
        params = self.net.param_pde_trainable
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        for param, grad in zip(params, grads):
            param.grad = grad


    
    def config_train(self, traintype = 'vanilla-inv', lr_options = None):
        self.traintype = traintype

        if traintype == 'vanilla-init' or traintype == 'vanilla-inv':
            # param_all include all parameters, including requires_grad = False (some pde parameter and embedding)
            self.optimizer['allparam'] = self.optim(self.net.param_all)
            self.ftrain = self.train_vanilla

        # other are new method
        else:
            optim_param_group = [
                {'params': self.net.param_net, 'lr': self.opts['lr_net']},
                {'params': self.net.param_pde_trainable, 'lr': self.opts['lr_pde']}
            ]
            self.optimizer['allparam'] = self.optim(optim_param_group,amsgrad=True)

            if traintype == 'adj-init' or traintype == 'adj-simu':
                # single optimizer for all parameters
                # learning rate for pde parameter is 0
                self.ftrain = self.train_simu
            elif traintype == 'adj-bi1opt':
                # single optimizer for all parameters, toggle pde_param lr between 0 and lr_pde
                self.ftrain = self.train_bilevel_singleopt
            else:
                raise ValueError(f'train type {traintype} not supported')

        if lr_options is None or lr_options['scheduler'] == 'constant':
            # constant learning rate
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer['allparam'], lr_lambda=lambda epoch: 1)

        elif lr_options['scheduler'] == 'cosine':
            T_max = self.opts['max_iter']
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer['allparam'], T_max=T_max, eta_min= self.opts['lr_pde']/10.0)
        else:
            raise ValueError(f'learning rate scheduler {lr_options["scheduler"]} not supported')

    def train(self):
        # move to device
        self.net.to(self.device)
        self.pde.dataset.to_device(self.device)

        print(f'train with: {self.traintype}')
        start = time.time()
        try:
            self.ftrain()
        except KeyboardInterrupt:
            print('Interrupted by user')
            self.info.update({'error':'interrupted by user'})
        except Exception as e:
            raise e
        
        # log training info
        if self.estop.epoch > 0:
            end = time.time()
            sec_per_step = (end - start) / self.estop.epoch
            mem =  get_mem_stats()
            self.info.update({'sec_per_step':sec_per_step})
            self.info.update(mem)
        
        self.logger.log_params(flatten(self.info))

    def train_simu(self):
        '''
        simultaneous training of network and pde parameter
        '''
        wloss_comp = {}
        epoch = 0
        
        while True:

            self.optimizer['allparam'].zero_grad()
            self.lossCollection.getloss()

            loss_net = self.lossCollection.get_wloss_sum(self.loss_net)
            loss_pde = self.lossCollection.get_wloss_sum(self.loss_pde)
            
            wloss_comp.update(self.lossCollection.wloss_comp)
            wloss_comp['lowertot'] = loss_net

            # monitor total loss
            stophere = self.estop(self.lossCollection.wtotal, self.net.params_dict, epoch)

            # print statistics at interval or at stop
            if epoch % self.opts['print_every'] == 0 or stophere:
                self.log_stat(wloss_comp, epoch)
                val = self.pde.validate(self.net)
                self.log_stat(val, epoch)
            if stophere:
                break
            
            # take gradient of data loss w.r.t pde parameter
            # self.set_grad(self.net.param_pde, self.lossCollection.wloss_comp['data'])
            self.set_pde_param_grad(loss_pde)
            
            # take gradient of residual loss w.r.t network parameter
            self.set_grad(self.net.param_net, loss_net)

            # 1 step of GD
            self.optimizer['allparam'].step()
            self.scheduler.step()
            epoch += 1

    
    def train_bilevel_singleopt(self):
        # single optimizer for all parameters, change learning rate manually

        def log_stat(wloss_comp, islower, epoch):
            # log statistics
            if epoch % self.opts['print_every'] == 0 or stophere:
                self.logger.log_metrics(wloss_comp, step=epoch)
                self.logger.log_metrics(self.net.params_dict, step=epoch)
                self.logger.log_metrics({'lower':islower}, step=epoch)
        epoch = 0
        wloss_comp = {}
        
        while True:

            self.optimizer['allparam'].zero_grad()
            self.lossCollection.getloss()

            loss_lower = self.lossCollection.get_wloss_sum(self.loss_net)
            loss_upper = self.lossCollection.wloss_comp['data']

            wloss_comp.update(self.lossCollection.wloss_comp)

            wloss_comp['lowertot'] = loss_lower
            wloss_comp['uppertot'] = loss_upper

            # check early stopping
            stophere = self.estop(wloss_comp['total'], self.net.params_dict, epoch)

            log_stat(wloss_comp, 0, epoch)


            if stophere:
                break  


            ### lower level
            epoch_lower = 0
            while loss_lower > self.opts['tol_lower']:
                # set second group learning rate to 0
                self.optimizer['allparam'].param_groups[1]['lr'] = 0.0

                # take gradient of residual loss w.r.t network parameter
                self.set_grad(self.net.param_net, loss_lower)
                self.optimizer['allparam'].step()

                epoch_lower += 1
                epoch += 1

                if epoch_lower == self.opts['max_iter_lower']:
                    break
                    
                
                # compute lower level loss
                self.optimizer['allparam'].zero_grad()
                self.lossCollection.getloss()
                wloss_comp.update(self.lossCollection.wloss_comp)

                loss_lower = self.lossCollection.get_wloss_sum(self.loss_net)
                wloss_comp['lowertot'] = loss_lower

                log_stat(wloss_comp, 1, epoch)
                val = self.pde.validate(self.net)
                self.log_stat(val, epoch)
            ### end of lower level

            if epoch_lower > 0:
                # redo upper level loss, since lower level has changed the network
                self.optimizer['allparam'].zero_grad()

                # reset learning rate
                self.optimizer['allparam'].param_groups[1]['lr'] = self.opts['lr_pde']
                self.lossCollection.getloss()
                wloss_comp.update(self.lossCollection.wloss_comp)
                
                loss_upper = self.lossCollection.wloss_comp['data']
                wloss_comp['uppertot'] = loss_upper
                
                loss_lower = self.lossCollection.get_wloss_sum(self.loss_net)
                wloss_comp['lowertot'] = loss_lower
                
                log_stat(wloss_comp, 0, epoch)
                val = self.pde.validate(self.net)
                self.log_stat(val, epoch)

            # take gradient of data loss w.r.t pde parameter
            self.set_pde_param_grad(self.lossCollection.wloss_comp['data'])
            
            # take gradient of residual loss w.r.t network parameter
            loss_lower = self.lossCollection.get_wloss_sum(self.loss_net)

            self.set_grad(self.net.param_net, loss_lower)
            
            # 1 step of GD for all net+pde params
            self.optimizer['allparam'].step()
            self.scheduler.step() # only step for allparam
            # next cycle
            epoch += 1

    def train_vanilla(self):
        '''
        vanilla training of network, update all parameters simultaneously
        '''

        epoch = 0
        while True:

            self.optimizer['allparam'].zero_grad()
            self.lossCollection.getloss()
            # check early stopping
            stophere = self.estop(self.lossCollection.wtotal, self.net.params_dict, epoch)

            # print statistics at interval or at stop
            if epoch % self.opts['print_every'] == 0 or stophere:
                val = self.pde.validate(self.net)
                self.logger.log_metrics(val, epoch)
                self.log_stat(self.lossCollection.wloss_comp, epoch)
            if stophere:
                break  

            self.lossCollection.wtotal.backward(retain_graph=True)
            self.optimizer['allparam'].step()
            self.scheduler.step()
            # next cycle
            epoch += 1


    def save_optimizer(self):
        # save optimizer
        traintype = self.traintype

        for key in self.optimizer.keys():
            fname = f"optimizer_{key}.pth"
            fpath = self.logger.gen_path(fname)
            torch.save(self.optimizer[key].state_dict(), fpath)
            print(f'save optimizer to {fpath}')
    

    def load_optim(self, optimizer, fpath):
        if not os.path.exists(fpath):
            print(f'optimizer file {fpath} not found, use default optimizer')
            return
        
        print(f'restore optimizer from {fpath}')
        optimizer.load_state_dict(torch.load(fpath))
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    
    def restore_optimizer(self, dirname):
        # restore optimizer, need dirname
        traintype = self.traintype
        if self.opts['reset_optim']:
            print('do not restore optimizer, reset optimizer to default')
            return

        for key in self.optimizer.keys():
            fname = f"optimizer_{key}.pth"
            fpath = os.path.join(dirname, fname)
            self.load_optim(self.optimizer[key], fpath)
            
            
            # adjust learning rate
            # for allparam, first is net, second is pde
            if key == 'allparam' and traintype.startswith('adj'):
                lr_net = self.optimizer[key].param_groups[0]['lr']
                lr_pde = self.optimizer[key].param_groups[1]['lr']
                self.optimizer[key].param_groups[0]['lr'] = self.opts['lr_net']
                self.optimizer[key].param_groups[1]['lr'] = self.opts['lr_pde']
                print(f'adjust lr_net from {lr_net} to {self.opts["lr_net"]}')
                print(f'adjust lr_pde from {lr_pde} to {self.opts["lr_pde"]}')
            

    def save_net(self):
        # save network
        
        net_path = self.logger.gen_path("net.pth")
        torch.save(self.net.state_dict(), net_path)
        print(f'save model to {net_path}')

    def restore_net(self, net_path):
        self.net.load_state_dict(torch.load(net_path))
        print(f'restore model from {net_path}')

    def save_dataset(self):
        ''' make prediction and save dataset'''
        self.pde.make_prediction(self.net)
        dataset_path = self.logger.gen_path("dataset.mat")
        self.pde.dataset.save(dataset_path)


    def save(self):
        '''saving dir from logger'''
        # save optimizer
        self.save_dataset()

        # if max_iter is 0, do not save optimizer and net
        if self.opts['max_iter']>0:
            self.save_optimizer()
            self.save_net()

    def restore(self, dirname):
        # restore optimizer
        self.restore_optimizer(dirname)

        fnet = os.path.join(dirname, 'net.pth')
        self.restore_net(fnet)

# simple test on training routine
# if __name__ == "__main__":

#     optobj = Options()
#     # change default for testing
#     optobj.opts['flags'] = 'smallrun,local'
#     optobj.opts['pde_opts']['problem'] = 'poisson'

#     optobj.parse_args(*sys.argv[1:])

     
#     device = set_device('cuda')
#     set_seed(0)

#     # setup pde
#     pde = create_pde_problem(**(optobj.opts['pde_opts']),datafile=optobj.opts['dataset_opts']['datafile'])
#     pde.print_info()

#     # setup logger
#     logger = Logger(optobj.opts['logger_opts']) 

#     # setup dataset
#     dataset = create_dataset_from_pde(pde, optobj.opts['dataset_opts'])

#     # setup network
#     net = DenseNet(**optobj.opts['nn_opts'],
#                                 output_transform=pde.output_transform,
#                                 params_dict=pde.param)

    
#     # set up los
#     lc = lossCollection(net, pde, optobj.opts['weights'])

#     print_dict(optobj.opts)
#     ### basic
    
#     trainer = Trainer(optobj.opts['train_opts'], net, pde, lc, logger)
    
#     trainer.config_train(optobj.opts['traintype'])

#     if optobj.opts['restore']:
#         trainer.restore(optobj.opts['restore'])

#     trainer.train()

#     trainer.save()






