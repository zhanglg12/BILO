import sys
import json
import ast


default_opts = {
    'logger_opts': {'use_mlflow':False,
                    'use_stdout':True,
                    'use_csv':False,
                    'experiment_name':'dev',
                    'run_name':'test',
                    'save_dir':'./runs/tmp'},
    'restore': '',
    'traintype': 'basic',
    'smallrun': False, 
    'device': 'cuda',
    'seed': 0,
    'transfer_opts':{
        'transfer_method': 'lora', # lora, freeze
        'nlayer_train': 1, # number of layers to train
        'rank': 4,
    },
    'pde_opts': {
        'problem': 'Simpleode',
    },
    'nn_opts': {
        'depth': 4,
        'width': 64,
        'input_dim': 1,
        'output_dim': 1,
        'use_resnet': False,
        'with_param': False,
        'useFourierFeatures':False,
    },
    'dataset_opts': {
        'N_res_train': 100,
        'N_res_test': 100,
        'N_dat_train': 100,
        'N_dat_test': 100,
        'datafile': '',
    },
    'train_opts': {
        'max_iter': 100000,
        'tolerance': 1e-6,
        'print_every': 20,
        'patience': 1000,
        'delta_loss':1e-5,
        'monitor_loss':True,
        'burnin':1000,
    },
    'lr' : 1e-3,
    'optimizer': 'adam',
    'adam_opts': {
    
    },
    'lbfgs_opts': {
        
    },
    'noise_opts':{
        'use_noise': False,
        'variance': 0.01,
        'length_scale': 0.0,
    },
    'weights': {
        'res': 1.0,
        'resgrad': 0.001,
        'data': 1.0,
        'paramgrad': None,
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
            # for vanilla PINN, nn does not include parameter
            self.opts['nn_opts']['with_param'] = False
            
            
        
        


if __name__ == "__main__":
    
    
    opts = Options()
    opts.parse_args(*sys.argv[1:])

    print (json.dumps(opts.opts, indent=2,sort_keys=True))