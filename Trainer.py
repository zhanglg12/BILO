# for training the network
# need: options, network, pde, dataset, lossCollection
from lossCollection import *
from DensePoisson import *
import mlflow

class Trainer:
    def __init__(self, opts, net, pde, dataset, lossCollection):
        self.opts = opts
        self.net = net
        self.pde = pde
        self.dataset = dataset
        self.lossCollection = lossCollection
        self.mlrun = None
        self.device = set_device()
    
    def setup_mlflow(self): 

        if self.opts['experiment_name'] == '':
            # do not use mlflow, self.mlrun is None
            return
        
        # end previous run if exist
        mlflow.end_run()
        # creat experiment if not exist
        mlflow.set_experiment(self.opts['experiment_name'])
        # start mlflow run
        mlflow.start_run(run_name=self.opts['run_name'])
        self.mlrun = mlflow.active_run()

        # mlflow.set_tracking_uri('')
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking uri: {tracking_uri}")

        mlflow.log_params(flatten(self.opts))

    def train(self):

        self.net.to(self.device)
        self.dataset.to_device(self.device)

        # print(self.device)

        epoch = 0
        estop = EarlyStopping(**self.opts['train_opts'])
        try:
            while True:
                total_loss = 0
                wloss_comp = {}

                for lossName in self.lossCollection:
                    lossObj = self.lossCollection[lossName]
                    
                    lossObj.getloss()
                    if lossName=='param':
                        if epoch % self.opts['train_opts']['param_every'] == 0:
                            lossObj.step()
                    else:
                        lossObj.step()

                    total_loss += lossObj.loss_val
                    wloss_comp.update(lossObj.wloss_comp)

                wloss_comp['total'] = total_loss

                # get weighted loss components
                for k in wloss_comp:
                    wloss_comp[k] = wloss_comp[k].item()

                # check early stopping
                stophere = estop(total_loss, {'D':self.net.D}, epoch)

                # print statistics at interval or at stop
                if epoch % self.opts['train_opts']['print_every'] == 0 or stophere:
                    print_statistics(epoch, **wloss_comp, D=self.net.D.item())
                    # log metric using mlflow if available
                    if self.mlrun is not None:
                        mlflow.log_metrics(wloss_comp, step=epoch)
                        mlflow.log_metrics({'D':self.net.D.item()}, step=epoch)
                if stophere:
                    break  

                # next cycle
                epoch += 1

        except KeyboardInterrupt:
            # if interrupted, exit training, save is called elsewhere
            print('Interrupted')

    def save(self, dirname):
        # save training results  
        
        def genpath(filename):
            path = os.path.join(dirname, filename)
            return path

        # save model and optimizer to mlflow
        torch.save(self.net.state_dict(), genpath("net.pth"))
        
        for lossObjName in self.lossCollection:
            lossObj = self.lossCollection[lossObjName]
            fname = f"optimizer_{lossObjName}.pth"
            fpath = genpath(fname)
            lossObj.save_state(fpath)
        
        # save options
        savedict(self.opts, genpath("options.json"))
        
        # save dataset
        self.dataset.save(genpath("dataset.mat"))
        
        print(f'artifacts saved {dirname}')
        
    

