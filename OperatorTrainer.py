#!/usr/bin/env python
# train a operator with data

import torch
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np

from DataSet import DataSet
from DeepONet import DeepONet, OpData
from Logger import Logger
from util import flatten

class OperatorTrainer:
    # operator learning
    def __init__(self, train_opts, weights, deeponet, olprob, dataset, device, logger:Logger):
        
        self.train_opts = train_opts
        self.weights = weights # weights for loss function
        self.deeponet = deeponet
        self.olprob = olprob
        self.dataset = dataset
        self.device = device
        self.logger = logger

        self.info = {}
        self.info['num_params'] = sum(p.numel() for p in self.deeponet.parameters())
        self.info['num_train_params'] = sum(p.numel() for p in self.deeponet.parameters() if p.requires_grad)

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
            # to device
            self.olprob.dataset.to_device(self.device)
            # for operator learning, first dimension is batch
            X_data, U_data = self.olprob.get_inverse_data()
            

            self.train_dataset = OpData(X_data, self.deeponet.pde_param, U_data)

            # set pde_param to be trainable
            self.deeponet.pde_param.requires_grad = True

            # if use residual loss, network parameters are also trainable
            if self.weights['res'] is not None:
                self.trainable_param = list(self.deeponet.parameters())
            else:
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
        
        self.logger.log_params(flatten(self.info))
        
    def save(self):
        # save data and network 
        self.save_net()
        
        if self.traintype == 'pretrain':
            # save prediction on all data
            self.save_data('pretrain_pred.mat')
        else:
            self.save_data('inverse_pred.mat')

            
        

    def train_loop(self):
        # train the operator with data
        
        max_iter = self.train_opts['max_iter']
        print_every = self.train_opts['print_every']
        batch_size = self.train_opts['batch_size']

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the optimizer and loss function
        self.optimizer = torch.optim.Adam(self.trainable_param, lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()

        # Start the training iterations
        step = 0
        while step < max_iter:
            for P_batch, U_batch in train_loader:
                # Training mode and forward pass
                self.deeponet.train()
                U_pred = self.deeponet(P_batch, self.train_dataset.X)

                total = 0

                loss_dat = self.loss_fn(U_pred, U_batch)
                loss_dat = self.weights['data'] * loss_dat
                metric = {'data': loss_dat.item()}
                total += loss_dat

                if self.weights['res'] is not None:
                    # Add residual loss
                    
                    # ad-hoc fix
                    if self.traintype == 'pretrain':
                        X_res= self.OpData.X
                    else:
                        # might be res or dat depending of use_res option
                        X_res= self.olprob.dataset['X_res_train']
                    
                    res_loss = self.olprob.residual_loss(self.deeponet, P_batch, X_res)
                    total += self.weights['res'] * res_loss
                    metric['res'] = res_loss.item()

                if self.traintype == 'inverse' and self.weights['l2grad'] is not None:
                    # Add regularization loss
                    # ad-hoc regularization loss, use for poivar only
                    regloss = self.olprob.regularization_loss(self.deeponet)
                    total += self.weights['l2grad'] * regloss
                    metric['l2grad'] = regloss.item()

                # Logging
                if step % print_every == 0 or step == max_iter - 1:
                    
                    if self.traintype == 'pretrain':
                        # for pretraining, evaluate on testing data
                        test_loss = self.evaluate()
                        metric['datatest'] = test_loss.item()
                    else:
                        # for inverse problem, evalute metrics w.r.t. exact solution
                        param_metric = self.olprob.get_metrics(self.deeponet)
                        metric.update(param_metric)

                        # ad-hoc regularization loss, use for poivar only
                        if self.weights['l2grad'] is not None:
                            metric['l2grad'] = regloss.item()

                    self.logger.log_metrics(metric, step=step)

                # Backpropagation
                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()

                # Increment the step count
                step += 1
                if step >= max_iter:
                    break
    
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
        if self.traintype == 'pretrain':
            self.olprob.make_prediction_pretrain(self.deeponet)
            fpath = self.logger.gen_path(filename)

            self.olprob.pred_dataset.save(fpath)
        
        elif self.traintype == 'inverse':
            self.olprob.make_prediction_inverse(self.deeponet)

            fpath = self.logger.gen_path(filename)
            self.olprob.dataset.save(fpath)

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