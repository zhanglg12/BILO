#!/usr/bin/env python
import sys
import os
from util import read_json

import torch
from OptionsOpLearn import OptionsOpLearn
from FKDeepONet import FKOperatorLearning
from OperatorTrainer import OperatorTrainer
from Logger import Logger
from MlflowHelper import load_artifact
from util import print_dict


opts = OptionsOpLearn()
opts.parse_args(*sys.argv[1:])




# restore from a previous run
if opts.opts['restore'] != '':
    # if the restore is a directory
    # restore the options, artifacts, and logger
    if os.path.isdir(opts.opts['restore']):
        opts_path = os.path.join(opts.opts['restore'], 'options.json')
        restore_opts = read_json(opts_path)
        restore_artifacts = {fname: os.path.join(opts.opts['restore'], fname) for fname in os.listdir(opts.opts['restore']) if fname.endswith('.pth')}
        path = opts.opts['restore']
        restore_artifacts['artifacts_dir'] = path
        print(f'restore from directory {path}')
    
    else:
        #  restore from exp_name:run_name
        name_str = opts.opts['restore']
        restore_opts, restore_artifacts = load_artifact(name_str=opts.opts['restore'])
        print(f'restore from mlflow {name_str}')
    
    # restore operator architecture
    opts.opts['nn_opts'].update(restore_opts['nn_opts'])
    # use same datafile
    opts.opts['datafile'] = restore_opts['datafile']


# setup the operator learning problem
fkoperator = FKOperatorLearning(datafile=opts.opts['datafile'], train_opts = opts.opts['train_opts'])
deeponet = fkoperator.setup_network(**opts.opts['nn_opts'])
logger = Logger(opts.opts['logger_opts'])

trainer = OperatorTrainer(opts.opts['train_opts'], deeponet, fkoperator, fkoperator.dataset, 'cuda', logger)

if opts.opts['restore'] != '':
    trainer.restore(restore_artifacts)

trainer.config_train(opts.opts['traintype'])

print_dict(opts.opts)
logger.log_options(opts.opts)
trainer.train()





