import torch
import numpy as np
import json

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

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)
        
def mse(x, y = 0):
    return torch.mean((x - y)**2)

def print_dict(d):
    print(json.dumps(d, indent=4,sort_keys=True))



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
def flatten(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)