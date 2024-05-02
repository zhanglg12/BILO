#!/usr/bin/env python
import torch
import numpy as np

from matplotlib import pyplot as plt

from DataSet import DataSet
from DeepONet import DeepONet, OpData
from BaseOperator import BaseOperator

class FKOperatorLearning(BaseOperator):
    def __init__(self, **kwargs):
        self.input_dim = 2
        self.output_dim = 1
        self.param_dim = 2
        self.dataset = DataSet(kwargs['datafile'])
        self.lambda_transform = lambda X, u: (0.5 * torch.sin(np.pi * X[:,1:2]) ** 2)+ u * X[:,1:2] * (1 - X[:,1:2]) * X[:,0:1]
    

    def setup_inverse(self):
        # setup inverse problem
        D0 = 1.0
        rho0 = 1.0
        self.pde_param = torch.nn.Parameter(torch.tensor([D0, rho0], dtype=torch.float32).reshape(1,2).to('cuda'))

        D_gt = 2.0
        rho_gt = 2.0
        U_data, X_data = self.get_inverse_data(D_gt, rho_gt, final_time=True)

        self.U_data = U_data.to('cuda')
        self.X_data = X_data.to('cuda')
        self.deeponet.to('cuda')
    
    def inverse_log(self, step):
        D = self.pde_param[0,0].item()
        rho = self.pde_param[0,1].item()
        print(f'step: {step}, D: {D}, rho: {rho}')

    def get_inverse_data(self, d, rho, final_time = False):
        # get the final time U at d, rho, and X
        tol = 1e-12
        x1 = np.abs(self.dataset['P'][:,0] - d) < tol
        x2 = np.abs(self.dataset['P'][:,1] - rho) < tol
        idx = np.where( x1 & x2)[0]

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

        U = U.reshape(nt,nx)
        U_final = U[:,-1]
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


    