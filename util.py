import torch
import numpy as np
import json
import inspect
import types


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, types.FunctionType):
            # for simple lambda function
            return inspect.getsource(obj)
        else:
            return super(MyEncoder, self).default(obj)

def savedict(dict, fpath):
    json.dump( dict, open( fpath, 'w' ), indent=4, cls=MyEncoder, sort_keys=True)

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)
        
def mse(x, y = 0):
    # mean square error
    return torch.mean((x - y)**2)

def print_dict(d):
    print(json.dumps(d, indent=4,cls=MyEncoder,sort_keys=True))

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_statistics(epoch, **losses):
    """
    Prints the epoch number and an arbitrary number of loss values.

    Args:
        epoch (int): The current epoch number.
        **losses (dict): A dictionary where the keys are descriptive names of the loss values,
                         and the values are the loss values themselves.
    """
    loss_strs = ', '.join(f'{name}: {value:.3e}' for name, value in losses.items())
    print(f'Epoch {epoch}, {loss_strs}')



# Function to set device priority: CUDA > MPS > CPU
def set_device(name = None):

    if name is not None:
        device = torch.device(name)
        print(f'Using device: {device}')
        return device

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


# https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
# turn a nested dict to a single dict
# def flatten(d, parent_key=''):
#     items = []
#     for k, v in d.items():
#         new_key = k
#         if isinstance(v, dict):
#             items.extend(flatten(v, new_key).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)

#
#  mlflow.log_params
#  Names may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/).
def flatten(nested_dict):
    """
    Flatten a nested dictionary into a single dictionary.
    For a nested dictionary, use key1.key2.....keyN as the concatenated key.

    Args:
    - nested_dict (dict): The nested dictionary to flatten.

    Returns:
    - dict: A flattened dictionary.
    """
    def flatten_helper(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_helper(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return flatten_helper(nested_dict)



def generate_grf(x, a, l):
    """
    Generate 1D Gaussian random field with mean 0, std a, and correlation length l.
    """
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


def to_double(x):
    """
    Converts a torch tensor to double
    if dictionary, convert each value to double
    """
    
    if isinstance(x, torch.Tensor):
        return x.item()
    elif isinstance(x, torch.nn.ParameterDict) or isinstance(x, dict):
        return {key: to_double(value) for key, value in x.items()}

    elif isinstance(x, int) or isinstance(x, float) or isinstance(x, str):
        # if native python type, return as is
        return x
        
    else:
        raise ValueError(f'Unknown type: {type(x)}')
    
        
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
