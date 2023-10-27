import torch
import torch.nn as nn
import torch.optim as optim
import os

from Options import *
from util import *

from DensePoisson import *

import mlflow




class Engine:
    def __init__(self, opts) -> None:

        self.opts = opts

        self.net = DensePoisson(**(self.opts['nn_opts'])).to(device)
        
        self.dataset = {}

        self.lossCollection = {}
    
    def setup_data(self):
        xtmp = torch.linspace(0, 1, 20).view(-1, 1)
        self.dataset['x_train_res'] = xtmp.to(device)
        self.dataset['x_train_res'].requires_grad_(True)

        # generate data, might be noisy
        if self.opts['traintype']=="init":
            # for init, use D_init, no noise
            self.dataset['u_data'] = self.net.u_exact(self.dataset['x_train_res'], self.net.init_D)
        else:
            # for basci/inverse, use D_exact
            self.dataset['u_data'] = self.net.u_exact(self.dataset['x_train_res'], self.opts['Dexact'])

            if self.opts['noise_opts']['use_noise']:
                self.dataset['noise'] = generate_grf(xtmp, self.opts['noise_opts']['variance'],self.opts['noise_opts']['length_scale'])
                self.dataset['u_data'] = self.dataset['u_data'] + self.dataset['noise'].to(device)

    def setup_lossCollection(self):

        if self.opts['traintype'] == 'basic':
            # no resgrad
            loss_pde_opts = {'weights':{'res':self.opts['weights']['res'],'data':self.opts['weights']['data']}}
            self.lossCollection['basic'] = lossCollection(self.net, self.dataset, list(self.net.parameters()), optim.Adam, loss_pde_opts)
        
        elif self.opts['traintype'] == 'init':
            # use all 3 losses
            loss_pde_opts = {'weights':{'res':self.opts['weights']['res'],'resgrad':self.opts['weights']['resgrad'],'data':self.opts['weights']['data']}}
            self.lossCollection['init'] = lossCollection(self.net, self.dataset, self.net.param_net, optim.Adam, loss_pde_opts)

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
        
        mlflow.start_run(run_name=self.opts['runname'])

        mlflow.set_tracking_uri('')
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking uri: {tracking_uri}")

        mlflow.log_params(flatten(self.opts))
        epoch = 0
        while True:
            # Zero the gradients
            total_loss = 0
            loss_comp = {}

            for lossName in self.lossCollection:
                lossObj = self.lossCollection[lossName]

                lossObj.getloss()
                lossObj.step()

                total_loss += lossObj.loss_val
                loss_comp.update(lossObj.loss_comp)

            loss_comp['total'] = total_loss

            # convert all value of loss_comp to float
            for k in loss_comp:
                loss_comp[k] = loss_comp[k].item()
            
            if epoch % self.opts['train_opts']['print_every'] == 0:
                print_statistics(epoch, **loss_comp, D=self.net.D.item())
                # log metric using mlflow
                mlflow.log_metrics(loss_comp, step=epoch)

            # Termination conditions
            # Exit the loop if loss is below tolerance or maximum iterations reached
            if loss_comp['total'] < self.opts['train_opts']['tolerance'] or epoch >= self.opts['train_opts']['max_iter']:
                break  
            
            epoch += 1  # Increment the epoch counter
        
        self.save()

    
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
            optim_path = os.path.join(artifact_dir,'artifacts',optim_fname)

            if os.path.exists(optim_path):
                lossObj.optimizer.load_state_dict(torch.load())
                print(f'optimizer {lossObjName} loaded from {optim_path}')
            else:
                print(f'optimizer {lossObjName} not found, use default optimizer')


    def save(self):
        
        # save model and optimizer to mlflow
        torch.save(self.net.state_dict(), "net.pth")
        mlflow.log_artifact("net.pth")

        for lossObjName in self.lossCollection:
            lossObj = self.lossCollection[lossObjName]
            torch.save(lossObj.optimizer.state_dict(), f"optimizer_{lossObjName}.pth")
            mlflow.log_artifact(f"optimizer_{lossObjName}.pth")

        with open("options.json", "w") as f:
            json.dump(self.opts, f)
        mlflow.log_artifact("options.json")
        artifact_uri = mlflow.get_artifact_uri()
        print(f"Artifact uri: {artifact_uri}")
        
        
        





if __name__ == "__main__":
    
    

    optobj = Options()
    optobj.parse_args(*sys.argv[1:])

    device = set_device(optobj.opts['device'])
    eng = Engine(optobj.opts)

    eng.setup_data()
    eng.setup_lossCollection()
    eng.run()

        