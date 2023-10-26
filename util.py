import torch
import numpy as np
import json
np.random.seed(7)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def savedict(dict, fpath):
    json.dump( dict, open( fpath, 'w' ), indent=4, cls=MyEncoder, sort_keys=True)

def mse(x, y = 0):
    return torch.mean((x - y)**2)


def print_statistics(epoch, **losses):
    """
    Prints the epoch number and an arbitrary number of loss values.

    Args:
        epoch (int): The current epoch number.
        **losses (dict): A dictionary where the keys are descriptive names of the loss values,
                         and the values are the loss values themselves.
    """
    loss_strs = ', '.join(f'{name}: {value:.3g}' for name, value in losses.items())
    print(f'Epoch {epoch}, {loss_strs}')



# Function to set device priority: CUDA > MPS > CPU
def set_device():
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: {device}')
    # Check for MPS (Metal) availability
    elif is_metal_available():
        device = torch.device('metal')
        print(f'Using device: {device}')
    # Default to CPU
    else:
        device = torch.device('cpu')
        print(f'Using device: {device}')

    return device




def generate_grf(x, a, l):
    # Ensure x is a torch tensor, if not, convert it to a torch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    # Convert x to a 1D numpy array for processing with numpy
    x_numpy = x.view(-1).numpy()

    # Meshgrid for covariance matrix calculation
    x1, x2 = np.meshgrid(x_numpy, x_numpy, indexing='ij')

    if abs(l) > 1e-6:
        # grf with length scale l
        K = a * np.exp(-0.5 * ((x1 - x2)**2) / (l**2))
        grf_numpy = np.random.multivariate_normal(mean=np.zeros_like(x_numpy), cov=K)
    else:
        # iid Gaussian
        grf_numpy = np.random.normal(loc=0.0, scale=a, size=x_numpy.shape)

    # Convert grf_numpy back to a torch tensor, reshaping to match the original shape of x
    grf_torch = torch.tensor(grf_numpy, dtype=torch.float32).view(x.shape)

    return grf_torch