#!/usr/bin/env python
import torch
import numpy as np

from matplotlib import pyplot as plt

from DataSet import DataSet
from DeepONet import DeepONet, OpData
from BaseOperator import BaseOperator

class FKOperatorLearning(BaseOperator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = 2
        self.output_dim = 1
        self.param_dim = 2
        self.lambda_transform = lambda X, u: (0.5 * torch.sin(np.pi * X[:,1:2]) ** 2)+ u * X[:,1:2] * (1 - X[:,1:2]) * X[:,0:1]

        self.D = 2.0
        self.rho = 2.0
    
    def get_metrics(self,pde_param):
        # take pde_param, tensor of trainable parameters
        # return dictionary of metrics
        return {'rD': pde_param[0,0].item(), 'rRHO': pde_param[0,1].item()}

    def setup_network(self, **kwargs):
        deeponet = DeepONet(param_dim=self.param_dim, X_dim=self.input_dim, output_dim=self.output_dim, **kwargs,
         lambda_transform=self.lambda_transform)

        D0 = 1.0
        rho0 = 1.0        
        deeponet.pde_param = torch.nn.Parameter(torch.tensor([D0, rho0], dtype=torch.float32).reshape(1,2).to('cuda'))

        return deeponet
    
    def inverse_log(self):
        D = self.pde_param[0,0].item()
        rho = self.pde_param[0,1].item()
        param_dict = {'rD': D, 'rRHO': rho}
        return param_dict

    def get_inverse_data(self):
        # get the final time U at d, rho, and X
        

        d = 2.0
        rho = 2.0
        final_time = True

        tol = 1e-12
        x1 = torch.abs(self.dataset['P'][:,0] - d) < tol
        x2 = torch.abs(self.dataset['P'][:,1] - rho) < tol
        idx = torch.where( x1 & x2)[0]

        # if idx is empty, return
        if len(idx) == 0:
            print('f{d}, {rho} not found in dataset')
            return
        
        U = self.dataset['U'][idx]
        if not final_time:
            X = self.dataset['X']
            return U, X
        
        # get the final time U
        nt = self.dataset['t'].numel()
        nx = self.dataset['x'].numel()

        U = U.reshape((nt,nx))
        # first dimension of U is batch
        U_final = U[:,-1].reshape((1,nx))
        # create X, first column 1, second column x coord
        X_final = torch.zeros(nx,2)
        X_final[:,1] = self.dataset['x']
        X_final[:,0] = self.dataset['t'][0,-1]
        
        return U_final, X_final

    def visualize(self, d, rho):
        # find the idx of P with d and rho
        
        self.dataset.to_np()
        tol = 1e-12
        x1 = np.abs(self.dataset['P'][:,0] - d) < tol
        x2 = np.abs(self.dataset['P'][:,1] - rho) < tol
        idx = np.where( x1 & x2)[0]

        # if idx is empty, return
        if len(idx) == 0:
            print('f{d}, {rho} not found in dataset')
            return

        U = self.dataset['U'][idx]

        p = torch.tensor([d, rho], dtype=torch.float32).reshape(1,2).to('cuda')

        U_pred = self.deeponet(p, self.OpData.X)
        U_pred = U_pred.cpu().detach().numpy()
        U_pred = U_pred.reshape(51,51)

        # reshape U to 2D
        U = U.reshape(51,51)
        
        # plot U, U_pred, and error
        fig, axs = plt.subplots(1,3, figsize=(15, 5))
        cax = axs[0].imshow(U)
        axs[0].set_title('U')
        fig.colorbar(cax, ax=axs[0])

        cax = axs[1].imshow(U_pred)
        axs[1].set_title('U_pred')
        fig.colorbar(cax, ax=axs[1])

        cax = axs[2].imshow(U-U_pred,cmap='plasma')
        axs[2].set_title('Error')
        fig.colorbar(cax, ax=axs[2])

        fig.tight_layout()
        
    


if __name__ == "__main__":
    import sys
    # test the OpDataSet class
    filename  = sys.argv[1]
    

    fkoperator = FKOperatorLearning(datafile=filename)

    fkoperator.init(batch_size=1000)
    fkoperator.train(max_iter=1000, print_every=100, save_every=5000)

    fkoperator.save_data('pred.mat')


    