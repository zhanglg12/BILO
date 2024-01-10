#!/usr/bin/env python

# for training the network
# need: options, network, pde, dataset, lossCollection
from lossCollection import *
from DensePoisson import *
from Logger import Logger
import torch.optim as optim

class Trainer:
    def __init__(self, opts, net, pde, dataset, lossCollection, logger:Logger):
        self.opts = opts
        self.logger = logger
        self.net = net
        self.pde = pde
        self.dataset = dataset
        self.lossCollection = lossCollection
        self.device = set_device()

        # early stopping
        self.estop = EarlyStopping(**self.opts)

        # move to device
        self.net.to(self.device)
        self.dataset.to_device(self.device)    

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
        if traintype == 'basic':
            param_to_train = self.net.param_all
            self.optimizer = optim.Adam(param_to_train, lr=self.opts['lr'])
            self.ftrain = self.train_vanilla
        elif traintype == 'simu':
            optim_param_group = [
                {'params': self.net.param_net, 'lr': self.opts['lr_net']},
                {'params': self.net.param_pde, 'lr': self.opts['lr_pde']}
            ]
            self.optimizer = optim.Adam(optim_param_group)
            self.ftrain = self.train_simu
        elif traintype == 'bilevel':
            optim_param_group = [
                {'params': self.net.param_net, 'lr': self.opts['lr_net']},
                {'params': self.net.param_pde, 'lr': self.opts['lr_pde']}
            ]
            self.optimizer_upper = optim.Adam(optim_param_group)
            self.optimizer_lower = optim.Adam(self.net.param_net, lr=self.opts['lr_net'])
            self.ftrain = self.train_bilevel
        else :
            raise ValueError(f'train type {traintype} not supported')

    def train(self):
        print(f'train with: {self.traintype}')
        try:
            self.ftrain()
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        pass


    def train_simu(self):
        '''
        simultaneous training of network and pde parameter
        '''
        epoch = 0
        while True:

            self.optimizer.zero_grad()
            self.lossCollection.getloss()

            # monitor total loss
            stophere = self.estop(self.lossCollection.wtotal, self.net.params_dict, epoch)

            # print statistics at interval or at stop
            if epoch % self.opts['print_every'] == 0 or stophere:
                self.log_stat(self.lossCollection.wloss_comp, epoch)
            if stophere:
                break  
            
            # take gradient of data loss w.r.t pde parameter
            self.set_grad(self.net.param_pde, self.lossCollection.wloss_comp['data'])
            
            # take gradient of residual loss w.r.t network parameter
            loss_net = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
            self.set_grad(self.net.param_net, loss_net)

            # 1 step of GD
            self.optimizer.step()
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

            self.optimizer_upper.zero_grad()
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
                loss_net = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
                self.set_grad(self.net.param_net, loss_net)
                
                self.optimizer_lower.step()

                epoch_lower += 1
                epoch += 1

                if epoch_lower == self.opts['max_iter_lower']:
                    break
                    
                
                # compute lower level loss
                self.lossCollection.getloss()
                wloss_comp.update(self.lossCollection.wloss_comp)
                wloss_comp['lowertot'] = loss_net

                log_stat(wloss_comp, 1, epoch)
            ### end of lower level

            if epoch_lower > 0:
                # redo upper level loss, since lower level has changed the network
                self.lossCollection.getloss()
                loss_upper = self.lossCollection.wloss_comp['data']
                wloss_comp['uppertot'] = loss_upper
                loss_lower = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
                wloss_comp['lowertot'] = loss_lower
                log_stat(wloss_comp, 0, epoch)

            
            # take gradient of data loss w.r.t pde parameter
            self.set_grad(self.net.param_pde, loss_upper)
            
            # take gradient of residual loss w.r.t network parameter
            loss_net = self.lossCollection.wloss_comp['res'] + self.lossCollection.wloss_comp['resgrad']
            self.set_grad(self.net.param_net, loss_net)
            
            # 1 step of GD for all net+pde params
            self.optimizer_upper.step()
            # next cycle
            epoch += 1

    def train_vanilla(self):
        '''
        vanilla training of network, update all parameters simultaneously
        '''

        epoch = 0
        while True:

            self.optimizer.zero_grad()
            self.lossCollection.getloss()
            self.lossCollection.wtotal.backward()
            self.optimizer.step()

            # check early stopping
            stophere = self.estop(self.lossCollection.wtotal, self.net.params_dict, epoch)

            # print statistics at interval or at stop
            if epoch % self.opts['print_every'] == 0 or stophere:
                self.log_stat(self.lossCollection.wloss_comp, epoch)
            if stophere:
                break  

            # next cycle
            epoch += 1


    def save_optimizer(self, dirname):
        # save optimizer
        traintype = self.traintype
        dirpath = os.path.join(RUNS, dirname)
        if self.traintype == 'bilevel':
            optim_name = self.optimizer_upper.__class__.__name__
            fname = f"optimizer_upper_{optim_name}.pth"
            fpath = os.path.join(dirpath, fname)
            torch.save(self.optimizer_upper.state_dict(), fpath)

            optim_name = self.optimizer_lower.__class__.__name__
            fname = f"optimizer_lower_{optim_name}.pth"
            fpath = os.path.join(dirpath, fname)
            torch.save(self.optimizer_lower.state_dict(), fpath)
            

        else:
            optim_name = self.optimizer.__class__.__name__
            fname = f"optimizer_{traintype}_{optim_name}.pth"
            fpath = os.path.join(dirpath, fname)
            torch.save(self.optimizer.state_dict(), fpath)
        
        print(f'save optimizer to {dirpath}')
            
    
    def restore_optimizer(self, dirname):
        # restore optimizer
        traintype = self.traintype
        dirpath = os.path.join(RUNS, dirname)
        if self.traintype == 'bilevel':
            optim_name = self.optimizer_upper.__class__.__name__
            fname = f"optimizer_upper_{optim_name}.pth"
            fpath = os.path.join(dirpath, fname)
            self.optimizer_upper.load_state_dict(torch.load(fpath))

            optim_name = self.optimizer_lower.__class__.__name__
            fname = f"optimizer_lower_{optim_name}.pth"
            fpath = os.path.join(dirpath, fname)
            self.optimizer_lower.load_state_dict(torch.load(fpath))
            
        else:
            optim_name = self.optimizer.__class__.__name__
            fname = f"optimizer_{traintype}_{optim_name}.pth"
            fpath = os.path.join(dirpath, fname)
            self.optimizer.load_state_dict(torch.load(fpath))
        
        print(f'restore optimizer from {dirpath}')
        


    def save(self, dirname):
        # save training results  
        # this change the device of the network
        dirpath = os.path.join(RUNS, dirname)
        # make directory if not exist
        os.makedirs(dirpath, exist_ok=True)

        # save optimizer
        self.save_optimizer(dirname)

        # save model and optimizer to mlflow
        net_path = os.path.join(dirpath,"net.pth")
        torch.save(self.net.state_dict(), net_path)
        print(f'save model to {net_path}')
        
    
        # save dataset
        dataset_path = os.path.join(dirpath, "dataset.mat")
        self.dataset.save(dataset_path)
    
        
                
    
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
    lc = lossCollection(net, pde, dataset, optobj.opts['weights'])

    print_dict(optobj.opts)
    ### basic
    
    trainer = Trainer(optobj.opts['train_opts'], net, pde, dataset, lc, logger)
    
    trainer.config_train(optobj.opts['traintype'])

    if optobj.opts['restore']:
        trainer.restore_optimizer(optobj.opts['restore'])

    trainer.train()

    trainer.save(optobj.opts['logger_opts']['save_dir'])






