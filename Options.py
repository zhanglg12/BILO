import sys
import json
import ast


default_opts = {
    'run_name': 'tmp',
    'experiment_name': 'dev',
    'restore': '',
    'traintype': 'basic',
    'smallrun': False, 
    'device': 'cuda',
    'nn_opts': {
        'depth': 3,
        'width': 64,
        'use_resnet': False,
        'basic': False,
        'init_D': 1.0,
        'p': 1
    },
    'train_opts': {
        'max_iter': 100000,
        'lr': 0.001,
        'tolerance': 1e-3,
        'print_every': 20,
    },
    'Dexact': 2.0,
    'noise_opts':{
        'use_noise': False,
        'variance': 0.01,
        'length_scale': 0.0,
    },
    'weights': {
        'res': 1.0,
        'resgrad': 1.0,
        'data': 1.0
    }
}


def update_nested_dict(nested_dict, key, value):
    """
    Recursively updates a nested dictionary by finding the specified key and assigning the new value to it.
    If the key is not found, the dictionary remains unchanged.
    If the key is found multiple times, a ValueError is raised.
    
    Args:
    - nested_dict (dict): the nested dictionary to update
    - key (str): the key to find and update
    - value (any): the new value to assign to the key
    
    Returns:
    - None
    """
    found =  0
    for k, v in nested_dict.items():
        if k == key:
            nested_dict[k] = value
            found += 1
        elif isinstance(v, dict):
            found +=update_nested_dict(v, key, value)

    if found>1:
        raise ValueError('Key %s found multiple times in dictionary' % key)

    return found

def copy_nested_dict(orig_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict):
            orig_dict[key] = copy_nested_dict(orig_dict.get(key, {}), value)
        else:
            orig_dict[key] = value
    return orig_dict


def get_nested_dict(nested_dict, target_key):
    """
    get value from nested dict, if not found, result is None
    """
    for k, v in nested_dict.items():
        if k == target_key:
            return v
        elif isinstance(v, dict):
            result = get_nested_dict(v, target_key)
            if result is not None:
                return result

class Options:
    def __init__(self):
        self.opts = default_opts
    
    def parse_args(self, *args):

        self.parse_nest_args(*args)
        self.processing()
    
    def parse_nest_args(self, *args):
        # parse args according to dictionary
        i = 0
        while i < len(args):
            key = args[i]
            default_val = get_nested_dict(self.opts, key)
            # if default_val is string, also save arg-value as string
            # otherwise, try to convert to other type
            if isinstance(default_val,str):
                val = args[i+1]
            else:
                try:
                    val = ast.literal_eval(args[i+1])
                except ValueError as ve:
                    print(f'error parsing {args[i]} and {args[i+1]}: {ve}')
                    sys.exit(1)
            
            found = update_nested_dict(self.opts, key, val)
            if found==0:
                raise ValueError('Key %s not found in dictionary' % key)
            i +=2
    
    def processing(self):
        if self.opts['smallrun']:
            # use small network for testing
            self.opts['nn_opts']['depth'] = 4
            self.opts['nn_opts']['width'] = 2
            self.opts['train_opts']['max_iter'] = 10
            self.opts['train_opts']['print_every'] = 1
        if self.opts['traintype'] == 'basic':
            self.opts['weights']['resgrad'] = 0.0
            
        
        


if __name__ == "__main__":
    
    
    opts = Options()
    opts.parse_args(*sys.argv[1:])

    print (json.dumps(opts.opts, indent=2,sort_keys=True))