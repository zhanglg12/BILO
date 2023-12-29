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
        self.mlrun = None
        self.device = set_device()
    
    
    def train(self):

        self.net.to(self.device)
        self.dataset.to_device(self.device)

        # print(self.device)

        epoch = 0
        estop = EarlyStopping(**self.opts)
        try:
            while True:
                total_loss = 0
                wloss_comp = {}

                for lossName in self.lossCollection:
                    lossObj = self.lossCollection[lossName]
                    
                    lossObj.getloss()
                    
                    lossObj.step()

                    total_loss += lossObj.loss_val
                    wloss_comp.update(lossObj.wloss_comp)

                wloss_comp['total'] = total_loss

                # get weighted loss components
                for k in wloss_comp:
                    wloss_comp[k] = wloss_comp[k].item()

                # check early stopping
                stophere = estop(total_loss, self.net.params_dict, epoch)

                # print statistics at interval or at stop
                if epoch % self.opts['print_every'] == 0 or stophere:
                    self.logger.log_metrics(wloss_comp, step=epoch)
                    self.logger.log_metrics(self.net.params_dict, step=epoch)
                if stophere:
                    break  

                # next cycle
                epoch += 1

        except KeyboardInterrupt:
            # if interrupted, exit training, save is called elsewhere
            print('Interrupted')

    def save(self, dirname):
        # save training results  
        # this change the device of the network
        
        def genpath(filename):
            path = os.path.join(dirname, filename)
            return path
        
        # make directory if not exist
        os.makedirs(dirname, exist_ok=True)

        # save model and optimizer to mlflow
        net_path = genpath("net.pth")
        torch.save(self.net.state_dict(), net_path)
        
        for lossObjName in self.lossCollection:
            lossObj = self.lossCollection[lossObjName]
            fname = f"optimizer_{lossObjName}.pth"
            fpath = genpath(fname)
            lossObj.save_state(fpath)
        
        
        # save dataset
        self.dataset.save(genpath("dataset.mat"))
        
        print(f'artifacts saved {dirname}')
        
    
# simple test on training routine
if __name__ == "__main__":
    device = set_device('cuda')

    # setup network
    param_dict = {'D':torch.tensor(1.0).to(device)}
    param_dict['D'].requires_grad = True
    net = DensePoisson(2,6, params_dict=param_dict, basic=True).to(device)

    # setup problem
    pde = PoissonProblem(p=1, exact_D=1.0)
    
    # setup dataset
    dataset = setup_dataset(pde,  {}, {'N_res_train':20, 'N_res_test':20})
    dataset['x_res_train'].requires_grad = True

    # setup logger
    logger_opts = {'use_mlflow':False, 'use_stdout':True, 'use_csv':False}
    logger = Logger(logger_opts)


    # set up loss
    param_to_train = net.param_all
    loss_pde_opts = {'weights':{'res': 1.0,'data': 1.0}}
    lc = {}
    lc['forward'] = lossCollection(net, pde, dataset, param_to_train, optim.Adam, loss_pde_opts)

    # set up trainer
    trainer_opts = {'print_every':1}
    trainer = Trainer(trainer_opts, net, pde, dataset, lc, logger)
    trainer.train()