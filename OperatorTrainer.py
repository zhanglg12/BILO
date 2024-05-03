#!/usr/bin/env python
# train a operator with data

import torch
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np

from DataSet import DataSet
from DeepONet import DeepONet, OpData
from Logger import Logger

class OperatorTrainer:
    # operator learning
    def __init__(self, train_opts, deeponet, olprob, dataset, device, logger:Logger):
        
        self.logger = logger
        self.deeponet = deeponet
        self.olprob = olprob
        self.dataset = dataset
        self.device = device
        self.train_opts = train_opts

    def init_pretrain_data(self):
        # For pre-training step, split the data into training and testing sets
        split = self.train_opts['split']

        self.OpData = OpData(self.dataset['X'], self.dataset['P'], self.dataset['U'])

        n_total = len(self.OpData)
        train_size = int(split * n_total)
        test_size = n_total - train_size

        print(f"Training size: {train_size}, Testing size: {test_size}")

        total_indices = torch.randperm(len(self.OpData))

        # Split indices into training and testing
        train_indices = total_indices[:train_size]
        test_indices = total_indices[train_size:]

        self.train_dataset = OpData(self.OpData.X, self.OpData.P[train_indices], self.OpData.U[train_indices])
        self.test_dataset = OpData(self.OpData.X, self.OpData.P[test_indices], self.OpData.U[test_indices])


    def config_train(self, traintype='pretrain'):
        # configure the training type
        self.traintype = traintype

        if traintype == 'pretrain':
            self.dataset.to_device(self.device)
            self.init_pretrain_data()
            self.trainable_param = self.deeponet.parameters()

        elif traintype == 'inverse':
            U_data, X_data = self.olprob.get_inverse_data()
            # to device
            U_data = U_data.to(self.device)
            X_data = X_data.to(self.device)

            self.train_dataset = OpData(X_data, self.deeponet.pde_param, U_data)
            self.deeponet.pde_param.requires_grad = True
            self.trainable_param = [self.deeponet.pde_param]

        else:
            raise ValueError('Invalid training type')

        self.deeponet.to(self.device)

    def train(self):
        try:
            self.train_loop()
        except KeyboardInterrupt:
            print('Interrupted by user')
        except Exception as e:
            raise e
        
        if self.traintype == 'pretrain':
            # save prediction on all data
            self.save_data('pretrain_pred.mat')

        self.save_net()
            
        

    def train_loop(self):
        # train the operator with data
        
        max_iter = self.train_opts['max_iter']
        print_every = self.train_opts['print_every']
        save_every = self.train_opts['save_every']

        # Initialize the optimizer and loss function
        self.optimizer = torch.optim.Adam(self.trainable_param, lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()

        # Start the training iterations
        step = 0
        while step <= max_iter:
            # Training mode and forward pass
            self.deeponet.train()
            U_pred = self.deeponet(self.train_dataset.P, self.train_dataset.X)
            loss = self.loss_fn(U_pred, self.train_dataset.U)

            # Logging
            if step % print_every == 0:
                metric = {'mse': loss.item()}
                param_metric = self.olprob.get_metrics(self.deeponet.pde_param)
                metric.update(param_metric)

                if self.traintype == 'pretrain':
                    test_loss = self.evaluate()
                    metric['testmse'] = test_loss.item()

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
        with torch.no_grad():
            U_pred = self.deeponet(self.test_dataset.P, self.test_dataset.X)
            loss = self.loss_fn(U_pred, self.test_dataset.U)
        
        return loss
    

    def load_checkpoint(self, checkpoint_path):
        
        # Load the state dict from the file
        state_dict = torch.load(checkpoint_path, map_location='cuda')

        # Load the state dict into the model
        self.deeponet.load_state_dict(state_dict)

        print(f"Model loaded from {checkpoint_path}")


    
    def save_data(self, filename):
        # save prediction on all data
        P = self.dataset['P']
        U_pred = self.deeponet(P, self.OpData.X)

        self.pred_dataset = DataSet()
        self.pred_dataset['U'] = U_pred
        
        fpath = self.logger.gen_path(filename)
        self.pred_dataset.save(fpath)

    def save_net(self):
        # save network
        net_path = self.logger.gen_path("net.pth")
        torch.save(self.deeponet.state_dict(), net_path)
        print(f'save model to {net_path}')

    def restore_net(self, net_path):
        self.deeponet.load_state_dict(torch.load(net_path))
        print(f'restore model from {net_path}')
    
    def restore(self, artifact_dict:dict):
        # restore the model
        # artifact_dict: dictionary of artifacts
        if 'net.pth' in artifact_dict:
            self.restore_net(artifact_dict['net.pth'])