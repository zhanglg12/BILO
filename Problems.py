# define problems for PDE
import torch

from FKproblem import FKproblem
from GBMproblem import GBMproblem
from PoissonProblem import PoissonProblem
from SimpleODEProblem import SimpleODEProblem
from LorenzProblem import LorenzProblem
from PoiVarProblem import PoiVarProblem
from HeatProblem import HeatProblem
from varFKproblem import varFKproblem

from PoissonProblem2 import PoissonProblem2

def create_pde_problem(pde_opts):
    problem_type = pde_opts['problem']
    if problem_type == 'poisson':
        return PoissonProblem(**pde_opts)
    if problem_type == 'poisson2':
        return PoissonProblem2(**pde_opts)
    elif problem_type == 'lorenz':
        return LorenzProblem(**pde_opts)
    elif problem_type == 'simpleode':
        return SimpleODEProblem(**pde_opts)
    elif problem_type == 'fk':
        return FKproblem(**pde_opts)
    elif problem_type == 'gbm':
        return GBMproblem(**pde_opts)
    elif problem_type == 'poivar':
        return PoiVarProblem(**pde_opts)
    elif problem_type == 'varfk':
        return varFKproblem(**pde_opts)
    elif problem_type == 'heat':
        return HeatProblem(**pde_opts)
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')

# if __name__ == "__main__":
    # simple visualization of the data set
    