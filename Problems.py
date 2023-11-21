# define problems for PDE
import torch
from DataSet import DataSet

class PoissonProblem():
    def __init__(self, **kwargs):
        super().__init__()
        self.p = kwargs['p']
        self.exact_D = kwargs['exact_D']
        

    def residual(self, nn, x, D):
        u_pred = nn.forward(x)
        u_x = torch.autograd.grad(u_pred, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = D * u_xx - self.f(x)
        # res_D = torch.autograd.grad(res, D, create_graph=True, grad_outputs=torch.ones_like(res))[0]
        return res, u_pred

    def f(self, x):
        return - (torch.pi * self.p)**2 * torch.sin(torch.pi * self.p * x)

    def u_exact(self, x, D):
        return torch.sin(torch.pi * self.p * x) / D


class PoissonProblem2():
    # u_xx + u_x = 1
    def __init__(self, **kwargs):
        super().__init__()
        self.p = kwargs['p']
        self.exact_D = kwargs['exact_D']
        

    def residual(self, nn, x, D):
        u_pred = nn.forward(x)
        u_x = torch.autograd.grad(u_pred, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = u_xx + u_x - 1.0
        return res, u_pred

    def f(self, x):
        return 0.0

    def u_exact(self, x, D):
        e = torch.exp(torch.tensor(1.0))
        return (e - torch.exp(1.0-x) + x - e * x) / (1.0 - e)

def create_pde_problem(**kwargs):
    problem_type = kwargs['problem']
    if problem_type == 'PoissonProblem':
        return PoissonProblem(**kwargs)
    elif problem_type == 'PoissonProblem2':
        return PoissonProblem2(**kwargs)
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')

