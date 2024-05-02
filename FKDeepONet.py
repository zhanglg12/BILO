#!/usr/bin/env python
import torch
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np

from matplotlib import pyplot as plt

from DataSet import DataSet
from DeepONet import DeepONet, OpData


class FKOperatorLearning():
    def __init__(self, **kwargs):
        self.input_dim = 2
        self.output_dim = 1
        self.param_dim = 2

        self.dataset = DataSet(kwargs['datafile'])
        
        self.lambda_transform = lambda X, u: (0.5 * torch.sin(np.pi * X[:,1:2]) ** 2)+ u * X[:,1:2] * (1 - X[:,1:2]) * X[:,0:1]
        

    def init_deeponet(self):
        # Initialize DeepONet
        self.deeponet = DeepONet(param_dim=self.param_dim, X_dim=self.input_dim, output_dim=self.output_dim,
         lambda_transform=self.lambda_transform)


    def init_dataloader(self, batch_size=100):
        # Initialize data loader
        n_total = len(self.OpData)
        train_size = int(0.8 * n_total)
        test_size = n_total - train_size

        total_indices = torch.randperm(len(self.OpData))

        # Split indices into training and testing
        train_indices = total_indices[:train_size]
        test_indices = total_indices[train_size:]

        train_dataset = Subset(self.OpData, train_indices)
        test_dataset = Subset(self.OpData, test_indices)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    def init(self, **kwargs):
        # setup network and dataloader
        batch_size = kwargs.get('batch_size', 100)

        # Initialize the network and dataloader
        self.init_deeponet()
        self.deeponet.to('cuda')

        self.dataset.to_device('cuda')
        self.OpData = OpData(self.dataset['X'], self.dataset['P'], self.dataset['U'])
        self.init_dataloader(batch_size=batch_size)
        
    def train(self, **kwargs):

        max_iter = kwargs.get('max_iter', 1000)
        print_every = kwargs.get('print_every', 100)
        save_every = kwargs.get('save_every', 500)

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
                    print(f'Step {step}, Loss: {loss.item()}, Test Loss: {test_loss}')

                # Checkpoint saving
                yes_save = (step>0) and (step % save_every == 0 or step == max_iter)
                if yes_save:
                    checkpoint_path = f'checkpoint_step_{step}.pth'
                    torch.save(self.deeponet.state_dict(), checkpoint_path)
                    print(f'Checkpoint saved to {checkpoint_path}')

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Increment the step count
                step += 1
    
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

    def inverse(self, final_time=False, max_iter=1000, print_every=100, save_every=5000):
        # solve the inverse problem
        D0 = 1
        rho0 = 1

        D_true = 2.0
        rho_true = 2.0

        p  = torch.tensor([D0, rho0], dtype=torch.float32).reshape(1,2).to('cuda')
        self.param = torch.nn.Parameter(p)

        # freeze the network
        for param in self.deeponet.parameters():
            param.requires_grad = False


        optimizer = torch.optim.Adam([self.param], lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        U_data, X_data = self.get_inverse_data(D_true, rho_true, final_time=final_time)

        X_data = X_data.to('cuda')
        U_data = U_data.to('cuda')

        step = 0
        while step <= max_iter:
            U_pred = self.deeponet(self.param, X_data)
            loss = loss_fn(U_pred, U_data)

            # Logging
            if step % print_every == 0 or step == max_iter:
                D = self.param.data[0,0].item()
                rho = self.param.data[0,1].item()
                loss_val = loss.item()
                print(f'Step {step}, D = {D:.4f}, rho = {rho:.4f}, Loss: {loss_val}')

            # Checkpoint saving
            # yes_save = (step>0) and (step % save_every == 0 or step == max_iter)
            # if yes_save:
            #     checkpoint_path = f'checkpoint_step_{step}.pth'
            #     print(f'Checkpoint saved to {checkpoint_path}')

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Increment the step count
            step += 1

    def evaluate(self):
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


    