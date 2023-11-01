#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import os

import numpy as np
from Options import *
from util import *

from DensePoisson import *

import mlflow
from MlflowHelper import MlflowHelper
from DataSet import DataSet

from PlotHelper import PlotHelper

from torchinfo import summary

def generate_grf(x, a, l):
    # Ensure x is a torch tensor, if not, convert it to a torch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    # Convert x to a 1D numpy array for processing with numpy
    x_numpy = x.view(-1).numpy()

    # Meshgrid for covariance matrix calculation
    x1, x2 = np.meshgrid(x_numpy, x_numpy, indexing='ij')

    if abs(l) > 1e-6:
        # grf with length scale l
        K = a * np.exp(-0.5 * ((x1 - x2)**2) / (l**2))
        grf_numpy = np.random.multivariate_normal(mean=np.zeros_like(x_numpy), cov=K)
    else:
        # iid Gaussian
        grf_numpy = np.random.normal(loc=0.0, scale=a, size=x_numpy.shape)

    # Convert grf_numpy back to a torch tensor, reshaping to match the original shape of x
    grf_torch = torch.tensor(grf_numpy, dtype=torch.float32).view(x.shape)

    return grf_torch

class Engine:
    def __init__(self, opts=None, run_id = None) -> None:

        self.device = set_device()

        if run_id is not None:
            self.load_from_id(run_id)
        else:
            self.opts = opts    
            self.net = DensePoisson(**(self.opts['nn_opts'])).to(self.device)
        
        self.dataset = {}
        self.lossCollection = {}
        self.mlrun = None
        self.artifact_dir = None
    
    def setup_data(self):
        dsopt = self.opts['dataset_opts']
        self.dataset = DataSet()
        xtmp = torch.linspace(0, 1, dsopt['N_res_train'] ).view(-1, 1)
        
        self.dataset['x_res_train'] = xtmp.to(self.device)
        self.dataset['x_res_train'].requires_grad_(True)

        self.dataset['x_res_test'] = torch.linspace(0, 1, dsopt['N_res_test']).view(-1, 1).to(self.device)

        # generate data, might be noisy
        if self.opts['traintype']=="init":
            # for init, use D_init, no noise
            self.dataset['u_res_train'] = self.net.u_exact(self.dataset['x_res_train'], self.net.init_D)
        else:
            # for basci/inverse, use D_exact
            self.dataset['u_res_train'] = self.net.u_exact(self.dataset['x_res_train'], self.opts['Dexact'])

            if self.opts['noise_opts']['use_noise']:
                self.dataset['noise'] = generate_grf(xtmp, self.opts['noise_opts']['variance'],self.opts['noise_opts']['length_scale'])
                self.dataset['u_res_train'] = self.dataset['u_res_train'] + self.dataset['noise'].to(self.device)
    
    def make_prediction(self):
        self.dataset['u_res_test'] = self.net(self.dataset['x_res_test'])
        

    def setup_lossCollection(self):

        if self.opts['traintype'] == 'basic':
            # no resgrad
            loss_pde_opts = {'weights':{'res':self.opts['weights']['res'],'data':self.opts['weights']['data']}}
            self.lossCollection['basic'] = lossCollection(self.net, self.dataset, list(self.net.parameters()), optim.Adam, loss_pde_opts)
        
        elif self.opts['traintype'] == 'init':
            # use all 3 losses
            loss_pde_opts = {'weights':{'res':self.opts['weights']['res'],
            'resgrad':self.opts['weights']['resgrad'],
            'data':self.opts['weights']['data'],
            'paramgrad':self.opts['weights']['paramgrad']}}
            self.lossCollection['pde'] = lossCollection(self.net, self.dataset, self.net.param_net, optim.Adam, loss_pde_opts)

        elif self.opts['traintype'] == 'inverse':
            loss_pde_opts = {'weights':{'res':self.opts['weights']['res'],'resgrad':self.opts['weights']['resgrad'],'data':self.opts['weights']['data']}}
            self.lossCollection['pde'] = lossCollection(self.net, self.dataset, self.net.param_net, optim.Adam, loss_pde_opts)

            loss_data_opts = {'weights':{'data':self.opts['weights']['data']}}
            self.lossCollection['data'] = lossCollection(self.net, self.dataset, self.net.param_pde, optim.Adam, loss_data_opts)
        else:
            raise ValueError(f'train type {self.opts["traintype"]} not supported')

        return 


    def run(self):

        self.load()

        print(json.dumps(self.opts, indent=2,sort_keys=True))
        
        # creat experiment if not exist
        mlflow.set_experiment(self.opts['experiment_name'])
        # start mlflow run
        mlflow.start_run(run_name=self.opts['run_name'])
        self.mlrun = mlflow.active_run()

        # mlflow.set_tracking_uri('')
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking uri: {tracking_uri}")

        mlflow.log_params(flatten(self.opts))
        epoch = 0

        estop = EarlyStopping(**self.opts['train_opts'])
        try:
            while True:
                # Zero the gradients
                total_loss = 0
                loss_comp = {}

                for lossName in self.lossCollection:
                    lossObj = self.lossCollection[lossName]
                    
                    lossObj.getloss()
                    if lossName=='param':
                        # update param every n-th iteration
                        if epoch % self.opts['train_opts']['param_every'] == 0:
                            lossObj.step()
                    else:
                        lossObj.step()

                    total_loss += lossObj.loss_val
                    loss_comp.update(lossObj.loss_comp)

                loss_comp['total'] = total_loss

                # convert all value of loss_comp to float
                for k in loss_comp:
                    loss_comp[k] = loss_comp[k].item()
                
                # check stop, print statistics
                stophere = estop(total_loss, {'D':self.net.D}, epoch)

                if epoch % self.opts['train_opts']['print_every'] == 0 or stophere:
                    print_statistics(epoch, **loss_comp, D=self.net.D.item())
                    # log metric using mlflow
                    mlflow.log_metrics(loss_comp, step=epoch)
                    mlflow.log_metrics({'D':self.net.D.item()}, step=epoch)

                epoch += 1  # Increment the epoch counter
                
                if stophere:
                    break  
                
        except KeyboardInterrupt:
            print('Interrupted')
            pass

        self.save()

    def load_from_id(self, run_id):
        helper = MlflowHelper()        
        artifact_paths = helper.get_artifact_paths(run_id)
        self.opts = read_json(artifact_paths['options.json'])
        
        # reconstruct net from options and load weight
        self.net = DensePoisson(**(self.opts['nn_opts'])).to(self.device)
        self.net.load_state_dict(torch.load(artifact_paths['net.pth']))
        print(f'net loaded from {artifact_paths["net.pth"]}')

    
    def load(self):
        # load model and optimizer from mlflow
        if self.opts['restore'] == '':
            return
        
        # artifact_dir is the path to the artifact folder
        # check if self.opts['restore'] is 32 string of numbers and letters
        if len(self.opts['restore'])==32:
            run_id = self.opts['restore']
            run = mlflow.get_run(run_id)
            artifact_dir = run.info.artifact_uri[7:] 
        # check if artifact_dir exists
        elif not os.path.exists(self.opts['restore']):
            artifact_dir = self.opts['restore']
        else:
            raise ValueError(f'artifact_dir {artifact_dir} not found')

        # note: might have issue if the net is not the same architecture
        # the net should be reconstructed from options.json
        # for now, just load the net
        net_path = os.path.join(artifact_dir,'net.pth')
        self.net.load_state_dict(torch.load(net_path))
        
        print(f'net loaded from {net_path}')
        for lossObjName in self.lossCollection:
            lossObj = self.lossCollection[lossObjName]
            
            optim_fname = f"optimizer_{lossObjName}.pth"
            optim_path = os.path.join(artifact_dir,optim_fname)

            if os.path.exists(optim_path):
                lossObj.optimizer.load_state_dict(torch.load(optim_path))
                print(f'optimizer {optim_fname} loaded from {optim_path}')
            else:
                print(f'optimizer {optim_fname} not found, use default optimizer')

    

    def save(self):
        
        artifact_uri = mlflow.get_artifact_uri()
        artifact_uri = artifact_uri[7:] # remove file://
        
        def genpath(filename):
            path = os.path.join(artifact_uri, filename)
            return path

        # save model and optimizer to mlflow
        torch.save(self.net.state_dict(), genpath("net.pth"))
        
        for lossObjName in self.lossCollection:
            lossObj = self.lossCollection[lossObjName]
            fname = f"optimizer_{lossObjName}.pth"
            fpath = genpath(fname)
            torch.save(lossObj.optimizer.state_dict(), fpath)
            
        # save options
        savedict(self.opts, genpath("options.json"))
        
        # save dataset
        self.dataset.save(genpath("dataset.mat"))
        
        print(f'artifacts saved {artifact_uri}')
        self.artifact_dir = artifact_uri


        
        
class EarlyStopping:
    def __init__(self,  **kwargs):
        self.tolerance = kwargs.get('tolerance', 1e-4)
        self.max_iter = kwargs.get('max_iter', 10000)
        self.patience = kwargs.get('patience', 100)
        self.delta_loss = kwargs.get('delta_loss', 0)
        self.delta_params = kwargs.get('delta_params', 0.001)
        self.burnin = kwargs.get('burnin',1000 )
        self.monitor_loss = kwargs.get('monitor_loss', True)
        self.monitor_params = kwargs.get('monitor_params', True)
        self.best_loss = None
        self.prev_params = None
        self.counter_param = 0
        self.counter_loss = 0

    def __call__(self, loss, params, epoch):
        if epoch >= self.max_iter:
            print('Stop due to max iteration')
            return True
        
        if loss < self.tolerance:
            print('Stop due to loss tolerance')
            return True
         
        if epoch < self.burnin:
            return False

        if self.monitor_loss:
            if self.best_loss is None:
                self.best_loss = loss
            elif loss > self.best_loss - self.delta_loss:
                self.counter_loss += 1
                if self.counter_loss >= self.patience:
                    print(f'Stop due to loss patience, best loss {self.best_loss}')
                    return True
            else:
                self.best_loss = loss
                self.counter_loss = 0

        # disabled for now, The difference is just leanring rate.
        # if self.monitor_params:
        #     if self.prev_params is None:
        #         self.prev_params =  {k:v.clone() for k,v in params.items()}
        #     else:
        #         param_diff = sum(torch.norm(params[k] - self.prev_params[k]) for k in params.keys())
        #         print(param_diff)
        #         self.prev_params = {k:v.clone() for k,v in params.items()}
        #         if param_diff < self.delta_params:
        #             self.counter_param += 1
        #             if self.counter_param >= self.patience:
        #                 print('Stop due to parameter convergence')
        #                 return True
        #         else:
        #             self.counter_param = 0
            

        return False

if __name__ == "__main__":
    
    

    optobj = Options()
    optobj.parse_args(*sys.argv[1:])
    
    # set seed
    np.random.seed(optobj.opts['seed'])
    torch.manual_seed(optobj.opts['seed'])

    eng = Engine(optobj.opts)

    eng.setup_data()
    eng.setup_lossCollection()
    
    summary(eng.net)

    eng.run()

    eng.dataset.to_device(eng.device)

    ph = PlotHelper(eng.net, eng.dataset, yessave=True, save_dir=eng.artifact_dir, Dexact=eng.opts['Dexact'])
    ph.plot_prediction()
    ph.device = eng.device

    D = eng.net.D.item()
    ph.plot_variation([D-0.1, D, D+0.1])

    # save command to file
    f = open("commands.txt", "a")
    f.write(' '.join(sys.argv))
    f.write('\n')
    f.close()