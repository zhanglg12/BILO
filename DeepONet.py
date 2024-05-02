#!/usr/bin/env python
import sys
import torch
import torch.nn as nn

from DataSet import DataSet

class DeepONet(nn.Module):
    def __init__(self, 
    param_dim=1, X_dim=1, output_dim=1,
    width=64,  branch_depth=4, trunck_depth=4,
    lambda_transform=lambda x, u: u):

        super(DeepONet, self).__init__()
        # param_dim is the dimension of the PDE parameter space
        # X_dim is the dimension of the PDE domain

        # branch net is a neural network that processes the parameter set
        # trunk net is a neural network that processes the coordinates
        
        self.width = width
        self.param_dim = param_dim
        self.X_dim = X_dim
        self.output_dim = output_dim
        self.branch_depth = branch_depth
        self.trunck_depth = trunck_depth
        self.lambda_transform = lambda_transform

        
        self.branch_net = self.build_subnet(param_dim, branch_depth)
        self.trunk_net = self.build_subnet(X_dim, trunck_depth)
        

    def build_subnet(self, input_dim, depth):

        layers = [input_dim] + [self.width]*depth  # List of layer widths

        layers_list = []
        for i in range(len(layers) - 1):
            layers_list.append(nn.Linear(layers[i], layers[i+1]))  # Add linear layer
            layers_list.append(nn.Tanh())  # Add activation layer
        return nn.Sequential(*layers_list)

    def forward(self, P_input, X_input):
        # Process each parameter set in the branch network
        branch_out = self.branch_net(P_input)  # [batch_size, width]


        # Process the fixed grid in the trunk network
        trunk_out = self.trunk_net(X_input)  # [num_points, width]

        # Compute the output as num_points x batch_size
        output = torch.mm(trunk_out, branch_out.t())  # [num_points, batch_size]

        # Apply the lambda transform to each batch of output
        output = self.lambda_transform(X_input, output)  # [num_points, batch_size]

        # tranpose the output to batch_size x num_points
        output = output.t()

        return output

# Note on output: The final `.squeeze()` is used to remove any unnecessary dimensions if `output_dim` is 1.

# operator learning dataset
class OpData(torch.utils.data.Dataset):
    #  N = gt*gx
    #  X = N-by-2 matrix of t and x
    #  U = m by N matrix of solutions
    #  P = m by 2 matrix of parameters
    def __init__(self, X, P, U):

        self.X = X
        self.P = P
        self.U = U
        
    def __len__(self):
        return self.P.shape[0]

    def __getitem__(self, idx):
        # get the idx-th item of P, X, and U
        
        return self.P[idx], self.U[idx]

# simple test of the network
if __name__ == "__main__":
    import sys
    # test the FKDataSet class
    filename  = sys.argv[1]
    

    dataset = DataSet(filename)
    
    dataset.printsummary()

    fkdata = FKData( dataset['X'], dataset['P'], dataset['U'])
    
    data_loader = torch.utils.data.DataLoader(fkdata, batch_size=10, shuffle=True)

    deeponet = DeepONet(2, 2, 1)

    # test dimension
    for data in data_loader:
        P, U = data
        u = deeponet(P, fkdata.X)
        print(u.shape)
        print(U.shape)
        break
