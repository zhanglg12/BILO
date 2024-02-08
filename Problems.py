# define problems for PDE
import torch

from FKproblem import FKproblem
from GBMproblem import GBMproblem
from PoissonProblem import PoissonProblem
from SimpleODEProblem import SimpleODEProblem
from LorenzProblem import LorenzProblem
from PoiVarProblem import PoiVarProblem

def create_pde_problem(**kwargs):
    problem_type = kwargs['problem']
    if problem_type == 'poisson':
        return PoissonProblem(**kwargs)
    elif problem_type == 'lorenz':
        return LorenzProblem(**kwargs)
    elif problem_type == 'simpleode':
        return SimpleODEProblem(**kwargs)
    elif problem_type == 'fk':
        return FKproblem(**kwargs)
    elif problem_type == 'gbm':
        return GBMproblem(**kwargs)
    elif problem_type == 'poivar':
        return PoiVarProblem(**kwargs)
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')



# if __name__ == "__main__":
    # simple visualization of the data set
    