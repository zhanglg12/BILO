#!/usr/bin/env python

# for training the network
# need: options, network, pde, dataset, lossCollection
from lossCollection import *
from DensePoisson import *
from Logger import Logger
import torch.optim as optim

class Trainer:
    def __init__(self, opts, net, pde, lossCollection, logger:Logger):
        self.opts = opts
        self.logger = logger
        self.net = net
        self.pde = pde
        
        self.lossCollection = lossCollection
        self.device = set_device()
        self.optimizer = {}

        # early stopping
        self.estop = EarlyStopping(**self.opts)

        

    def log_stat(self, wloss_comp, epoch,):
        # log statistics
        self.logger.log_metrics(wloss_comp, step=epoch)
        self.logger.log_metrics(self.net.params_dict, step=epoch)
    
    def set_grad(self, params, loss):
        '''
        set gradient of loss w.r.t params
        '''
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        for param, grad in zip(params, grads):
            param.grad = grad
    
    def config_train(self, traintype = 'basic'):
        self.traintype = traintype
        if traintype == 'basic' or traintype == 'fwd':
            param_to_train = self.net.param_all
            self.optimizer['allparam'] = optim.Adam(param_to_train, lr=self.opts['lr'])
            self.ftrain = self.train_vanilla
        
        elif traintype == 'adj-init':
            optim_param_group = [
                {'params': self.net.param_net, 'lr': self.opts['lr_net']},
                {'params': self.net.param_pde, 'lr': 0.0}
            ]
            self.optimizer['allparam'] = optim.Adam(optim_param_group)
            self.ftrain = self.train_simu
            

        elif traintype == 'adj-simu':
            optim_param_group = [
                {'params': self.net.param_net, 'lr': self.opts['lr_net']},
                {'params': self.net.param_pde, 'lr': self.opts['lr_pde']}
            ]
            self.optimizer['allparam'] = optim.Adam(optim_param_group, amsgrad=True)
            self.ftrain = self.train_simu
        elif traintype == 'adj-bi':
            optim_param_group = [
                {'params': self.net.param_net, 'lr': self.opts['lr_net']},
                {'params': self.net.param_pde, 'lr': self.opts['lr_pde']}
            ]
            # all parameters
            self.optimizer['allparam'] = optim.Adam(optim_param_group,amsgrad=True)
            # only network parameter
            self.optimizer['netparam'] = optim.Adam(self.net.param_net, lr=self.opts['lr_net'],amsgrad=True)
            self.ftrain = self.train_bilevel
        elif traintype == 'adj-bi1opt':
            optim_param_group = [
                {'params': self.net.param_net, 'lr': self.opts['lr_net']},
                {'params': self.net.param_pde, 'lr': self.opts['lr_pde']}
            ]
            # all parameters
            self.optimizer['allparam'] = optim.Adam(optim_param_group,amsgrad=True)
            self.ftrain = self.train_bilevel_singleopt
        else :
            raise ValueError(f'train type {traintype} not supported')

    def train(self):
        # move to device
        self.net.to(self.device)
        self.pde.dataset.to_device(self.device)

        print(f'train with: {self.traintype}')
        try:
            self.ftrain()
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        


    def train_simu(self):
        '''
        simultaneous training of network and pde parameter
        '''
        wloss_comp = {}
        epoch = 0
        while True:

            self.optimizer['allparam'].zero_grad()
            self.lossCollection.getloss()

            loss_net = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
            wloss_comp.update(self.lossCollection.wloss_comp)
            wloss_comp['lowertot'] = loss_net

            # monitor total loss
            stophere = self.estop(self.lossCollection.wtotal, self.net.params_dict, epoch)

            # print statistics at interval or at stop
            if epoch % self.opts['print_every'] == 0 or stophere:
                self.log_stat(wloss_comp, epoch)
            if stophere:
                break  
            
            # take gradient of data loss w.r.t pde parameter
            self.set_grad(self.net.param_pde, self.lossCollection.wloss_comp['data'])
            
            # take gradient of residual loss w.r.t network parameter
            self.set_grad(self.net.param_net, loss_net)

            # 1 step of GD
            self.optimizer['allparam'].step()
            epoch += 1


    def train_bilevel(self):

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
            self.optimizer['netparam'].zero_grad()
            self.lossCollection.getloss()

            loss_lower = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
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

                # take gradient of residual loss w.r.t network parameter
                self.set_grad(self.net.param_net, loss_lower)
                self.optimizer['netparam'].step()

                epoch_lower += 1
                epoch += 1

                if epoch_lower == self.opts['max_iter_lower']:
                    break
                    
                
                # compute lower level loss
                self.optimizer['netparam'].zero_grad()
                self.lossCollection.getloss()

                loss_lower = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
                
                wloss_comp.update(self.lossCollection.wloss_comp)
                wloss_comp['lowertot'] = loss_lower

                log_stat(wloss_comp, 1, epoch)
            ### end of lower level

            if epoch_lower > 0:
                # redo upper level loss, since lower level has changed the network
                self.optimizer['allparam'].zero_grad()
                self.lossCollection.getloss()
                loss_upper = self.lossCollection.wloss_comp['data']
                wloss_comp['uppertot'] = loss_upper
                loss_lower = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
                wloss_comp['lowertot'] = loss_lower
                log_stat(wloss_comp, 0, epoch)

            # take gradient of data loss w.r.t pde parameter
            self.set_grad(self.net.param_pde, loss_upper)
            
            # take gradient of residual loss w.r.t network parameter
            loss_lower = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
            self.set_grad(self.net.param_net, loss_lower)
            
            # 1 step of GD for all net+pde params
            self.optimizer['allparam'].step()
            # next cycle
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

            loss_lower = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
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

                loss_lower = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
                
                wloss_comp.update(self.lossCollection.wloss_comp)
                wloss_comp['lowertot'] = loss_lower

                log_stat(wloss_comp, 1, epoch)
            ### end of lower level

            if epoch_lower > 0:
                # redo upper level loss, since lower level has changed the network
                self.optimizer['allparam'].zero_grad()

                # reset learning rate
                self.optimizer['allparam'].param_groups[1]['lr'] = self.opts['lr_pde']
                self.lossCollection.getloss()
                loss_upper = self.lossCollection.wloss_comp['data']
                wloss_comp['uppertot'] = loss_upper
                loss_lower = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
                wloss_comp['lowertot'] = loss_lower
                log_stat(wloss_comp, 0, epoch)

            # take gradient of data loss w.r.t pde parameter
            self.set_grad(self.net.param_pde, loss_upper)
            
            # take gradient of residual loss w.r.t network parameter
            loss_lower = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
            self.set_grad(self.net.param_net, loss_lower)
            
            # 1 step of GD for all net+pde params
            self.optimizer['allparam'].step()
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
            self.lossCollection.wtotal.backward(retain_graph=True)
            self.optimizer['allparam'].step()

            # check early stopping
            stophere = self.estop(self.lossCollection.wtotal, self.net.params_dict, epoch)

            # print statistics at interval or at stop
            if epoch % self.opts['print_every'] == 0 or stophere:
                self.log_stat(self.lossCollection.wloss_comp, epoch)
            if stophere:
                break  

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
            
        optimizer.load_state_dict(torch.load(fpath))
        print(f'restore optimizer from {fpath}')
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    
    def restore_optimizer(self, dirname):
        # restore optimizer, need dirname
        traintype = self.traintype

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
        # save dataset
        self.pde.make_prediction(self.net)
        dataset_path = self.logger.gen_path("dataset.mat")
        self.pde.dataset.save(dataset_path)


    def save(self):
        '''saving dir from logger'''
        # save optimizer
        self.save_optimizer()
        self.save_net()
        self.save_dataset()

    def restore(self, dirname):
        # restore optimizer
        self.restore_optimizer(dirname)

        fnet = os.path.join(dirname, 'net.pth')
        self.restore_net(fnet)
        
                
    
# simple test on training routine
if __name__ == "__main__":

    optobj = Options()
    # change default for testing
    optobj.opts['flags'] = 'smallrun,local'
    optobj.opts['pde_opts']['problem'] = 'poisson'

    optobj.parse_args(*sys.argv[1:])

     
    device = set_device('cuda')
    set_seed(0)

    # setup pde
    pde = create_pde_problem(**(optobj.opts['pde_opts']),datafile=optobj.opts['dataset_opts']['datafile'])
    pde.print_info()

    # setup logger
    logger = Logger(optobj.opts['logger_opts']) 

    # setup dataset
    dataset = create_dataset_from_pde(pde, optobj.opts['dataset_opts'])

    # setup network
    net = DensePoisson(**optobj.opts['nn_opts'],
                                output_transform=pde.output_transform,
                                params_dict=pde.param)

    
    # set up los
    lc = lossCollection(net, pde, optobj.opts['weights'])

    print_dict(optobj.opts)
    ### basic
    
    trainer = Trainer(optobj.opts['train_opts'], net, pde, lc, logger)
    
    trainer.config_train(optobj.opts['traintype'])

    if optobj.opts['restore']:
        trainer.restore(optobj.opts['restore'])

    trainer.train()

    trainer.save()






