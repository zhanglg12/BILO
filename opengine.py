#!/usr/bin/env python
import sys
import os
from datetime import datetime
from util import read_json, print_dict

import torch

from OptionsOpLearn import OptionsOpLearn
from OperatorTrainer import OperatorTrainer
from Logger import Logger
from MlflowHelper import load_artifact

from FKDeepONet import FKOperatorLearning
from VarPoiDeepONet import VarPoiDeepONet


opts = OptionsOpLearn()
opts.parse_args(*sys.argv[1:])

pid = os.getpid()
# save command to file
f = open("commands.txt", "a")
f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
f.write(f'  pid: {pid}')
f.write('\n')
f.write(' '.join(sys.argv))
f.write('\n')
f.close()


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


# setup the operator learning problem
if opts.opts['pde_opts']['problem'] == 'fk':
    fkoperator = FKOperatorLearning(**opts.opts['pde_opts'])
elif opts.opts['pde_opts']['problem'] == 'varpoi':
    fkoperator = VarPoiDeepONet(**opts.opts['pde_opts'])
else:
    raise ValueError(f"problem {opts.opts['pde_opts']['problem']} not recognized")

# setup dataset for training
if opts.opts['traintype'] == 'inverse':
    fkoperator.setup_dataset(opts.opts['dataset_opts'], opts.opts['noise_opts'])

# setup netowrk
deeponet = fkoperator.setup_network(**opts.opts['nn_opts'])
logger = Logger(opts.opts['logger_opts'])

trainer = OperatorTrainer(opts.opts['train_opts'], deeponet, fkoperator, fkoperator.dataset, 'cuda', logger)

if opts.opts['restore'] != '':
    trainer.restore(restore_artifacts)

trainer.config_train(opts.opts['traintype'])

print_dict(opts.opts)
logger.log_options(opts.opts)
trainer.train()
trainer.save()

if opts.opts['traintype'] == 'inverse':
    fkoperator.visualize(savedir=logger.get_dir())



