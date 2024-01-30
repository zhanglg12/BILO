# define problems for PDE
import torch
from DataSet import DataSet


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from util import generate_grf

from FKproblem import FKproblem
from GBMproblem import GBMproblem
from PoissonProblem import PoissonProblem
from SimpleODEProblem import SimpleODEProblem
from LorenzProblem import LorenzProblem


def create_pde_problem(**kwargs):
    problem_type = kwargs['problem']
    if problem_type == 'poisson':
        return PoissonProblem(**kwargs)
    elif problem_type == 'poisson2':
        return PoissonProblem2(**kwargs)
    elif problem_type == 'lorenz':
        return LorenzProblem(**kwargs)
    elif problem_type == 'simpleode':
        return SimpleODEProblem(**kwargs)
    elif problem_type == 'fk':
        return FKproblem(**kwargs)
    elif problem_type == 'gbm':
        return GBMproblem(**kwargs)
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')




def add_noise(dataset, noise_opts):
    '''
    add noise to each coordinate of u_dat_train
    For now, assuming x is 1-dim, u is d-dim. 
    For ODE problem, this means the noise is correlated in time (if add length scale)
    '''
    dim = dataset['u_dat_train'].shape[1]
    x = dataset['x_dat_train'] # (N,1)
    noise = torch.zeros_like(dataset['u_dat_train'])
    for i in range(dim):
        tmp = generate_grf(x, noise_opts['variance'], noise_opts['length_scale'])
        noise[:,i] = tmp.squeeze()

    dataset['noise'] = noise
    dataset['u_dat_train'] = dataset['u_dat_train'] + dataset['noise']
    
    return dataset

# if __name__ == "__main__":
    # simple visualization of the data set
    