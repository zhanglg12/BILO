#!/usr/bin/env python
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np

from DataSet import DataSet
from DeepONet import DeepONet, OpData
from Logger import Logger

class BaseOperator(ABC):
    # operator learning
    def __init__(self, **kwargs):
        self.input_dim = None
        self.output_dim = None
        self.param_dim = None
        self.dataset = DataSet(kwargs['datafile'])        
        self.lambda_transform = None


        # for inverse problem
        self.pde_param = None
        self.X_data = None
        self.U_data = None

        self.train_opts = kwargs['train_opts']
    

    def setup_logger(self, logger_opts):
        self.logger = Logger(logger_opts)

    def init_deeponet(self):
        # Initialize DeepONet
        self.deeponet = DeepONet(param_dim=self.param_dim, X_dim=self.input_dim, output_dim=self.output_dim,
         lambda_transform=self.lambda_transform)


    def init_dataloader(self):
        # Initialize data loader

        batch_size = self.train_opts['batch_size']
        split = self.train_opts['split'] # percentage of training data

        n_total = len(self.OpData)
        train_size = int(split * n_total)
        test_size = n_total - train_size

        total_indices = torch.randperm(len(self.OpData))

        # Split indices into training and testing
        train_indices = total_indices[:train_size]
        test_indices = total_indices[train_size:]

        train_dataset = Subset(self.OpData, train_indices)
        test_dataset = Subset(self.OpData, test_indices)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    def init_training(self):
        # setup network and dataloader before training
        # move to cuda
        # Initialize the network and dataloader
        
        self.deeponet.to('cuda')
        self.dataset.to_device('cuda')

        self.OpData = OpData(self.dataset['X'], self.dataset['P'], self.dataset['U'])
        self.init_dataloader()
    

    def pretrain(self, **kwargs):
        # train the operator with data

        max_iter = self.train_opts['max_iter']
        print_every = self.train_opts['print_every']
        save_every = self.train_opts['save_every']

        # Initialize the optimizer and loss function
        self.optimizer = torch.optim.Adam(self.deeponet.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()

        # Start the training iterations
        step = 0
        while step < max_iter:
            for P, U in self.train_loader:
                # Training mode and forward pass
                self.deeponet.train()
                U_pred = self.deeponet(P, self.OpData.X)
                loss = self.loss_fn(U_pred, U)

                # Logging
                if step % print_every == 0:
                    test_loss = self.evaluate()
                    metric = {'trainmse': loss.item(), 'testmse': test_loss}
                    self.logger.log_metrics(metric, step=step)

                # Checkpoint saving
                yes_save = (step>0) and (step % save_every == 0 or step == max_iter)
                if yes_save:
                    ckpt_name = f'checkpoint_step_{step}.pth'
                    ckpt_path = self.logger.gen_path(ckpt_name)
                    torch.save(self.deeponet.state_dict(), ckpt_path)
                    print(f'Checkpoint saved to {ckpt_path}')

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Increment the step count
                step += 1
    
    @abstractmethod
    def setup_inverse(self, **kwargs):
        # setup self.pde_param
        pass

    @abstractmethod
    def inverse_log(self):
        # return parameters for logging
        pass

    def inverse(self):
        # inverse problem

        max_iter = self.train_opts['max_iter']
        print_every = self.train_opts['print_every']
        
        # freeze the network
        self.deeponet.freeze()

        self.optimizer = torch.optim.Adam([self.pde_param], lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        step = 0
        while step <= max_iter:
            U_pred = self.deeponet(self.pde_param, self.X_data)
            loss = loss_fn(U_pred, self.U_data)

            # Logging
            if step % print_every == 0 or step == max_iter:
                metric = self.inverse_log()
                metric['loss'] = loss.item()
                self.logger.log_metrics(metric, step=step)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Increment the step count
            step += 1

    def evaluate(self):
        # evaluate on testing data
        self.deeponet.eval()
        total_loss = 0
        with torch.no_grad():
            for P, U in self.test_loader:
                U_pred = self.deeponet(P, self.OpData.X)
                loss = self.loss_fn(U_pred, U)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        return avg_loss
    

    def load_checkpoint(self, checkpoint_path):
        
        # Load the state dict from the file
        state_dict = torch.load(checkpoint_path, map_location='cuda')

        # Load the state dict into the model
        self.deeponet.load_state_dict(state_dict)

        print(f"Model loaded from {checkpoint_path}")


    
    def save_data(self, filename):
        # save data as mat
        P = self.dataset['P']
        U_pred = self.deeponet(P, self.OpData.X)

        self.pred_dataset = DataSet()
        self.pred_dataset['U'] = U_pred
        self.pred_dataset.to_np()
        self.pred_dataset.save(filename)


    