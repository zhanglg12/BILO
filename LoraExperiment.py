#!/usr/bin/env python

from util import *

from DensePoisson import *
from MlflowHelper import *
from DataSet import DataSet

# required by minlora
from minlora import *
from functools import partial

from matplotlib import pyplot as plt


from Options import *

from Trainer import *
from lossCollection import *
from Problems import *
from PlotHelper import PlotHelper

from torchinfo import summary


optobj = Options()
optobj.parse_args(*sys.argv[1:])

# set seed
np.random.seed(optobj.opts['seed'])
torch.manual_seed(optobj.opts['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(optobj.opts['seed'])

# load model
model, opt = load_model(name_str=optobj.opts['restore'])

optobj.opts['nn_opts'].update(opt['nn_opts'])

# do not use this function, it will add lora to all layers
# add_lora(model)

# this skip fflayer and fcD
if optobj.opts['transfer_opts']['transfer_method'] == 'lora':
    rk = optobj.opts['transfer_opts']['rank']
    lora_config = {
        nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=rk),
        },
    }
    model.freeze_layers_except(0)
    add_lora_by_name(model, ['input_layer','hidden_layers','output_layer'],lora_config = lora_config)

    param_to_train = list(get_lora_params(model))

elif optobj.opts['transfer_opts']['transfer_method'] == 'freeze':
    nlayer = optobj.opts['transfer_opts']['nlayer_train']
    model.freeze_layers_except(nlayer)

    param_to_train = model.param_net

# setup new problem
pde = create_pde_problem(**optobj.opts['pde_opts'])
dataset = setup_dataset(pde,  optobj.opts['noise_opts'], optobj.opts['dataset_opts'])
loss_pde_opts = {'weights':{'res':optobj.opts['weights']['res'],'data':optobj.opts['weights']['data']}}
lc = {}
lc['forward'] = lossCollection(model, pde, dataset, param_to_train, optim.Adam, loss_pde_opts)



dataset['x_res_train'].requires_grad = True

estop = EarlyStopping(**optobj.opts['train_opts'])

trainer = Trainer(optobj.opts, model, pde, dataset, lc)

summary(model,verbose=2,col_names=["num_params", "trainable"])

trainer.setup_mlflow()
trainer.train()


artifact_dir = get_artifact_dir()


# if use lora, save the low rank state dict, merge weight (can not be undone)
# if freeze, the state dict is already saved

if optobj.opts['transfer_opts']['transfer_method'] == 'lora':
    lora_state_dict = get_lora_state_dict(model)
    torch.save(lora_state_dict, get_artifact_path('lora_state_dict.pth'))
    merge_lora(model)

    trainer.save(artifact_dir)# need to merge the weight so that the prediction is correct

elif optobj.opts['transfer_opts']['transfer_method'] == 'freeze':
    trainer.save(artifact_dir)

else:
    raise ValueError(f'Unknown transfer transfer_method: {optobj.opts["transfer_opts"]["transfer_method"]}')

dataset.to_device(trainer.device)

ph = PlotHelper(pde, dataset, yessave=True, save_dir=artifact_dir, exact_D=pde.exact_D)
ph.plot_prediction(model)