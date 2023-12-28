#!/usr/bin/env python
from Engine import *

optobj = Options()
optobj.parse_args(*sys.argv[1:])

# set seed
set_seed(optobj.opts['seed'])

eng = Engine(optobj.opts)

eng.setup_problem()
eng.setup_network()
eng.setup_logger()
eng.setup_data()
eng.dataset.to_device(eng.device)
eng.setup_lossCollection()

summary(eng.net)

eng.run()

eng.dataset.to_device(eng.device)



ph = PlotHelper(eng.pde, eng.dataset, yessave=True, save_dir=eng.logger.get_dir())
ph.plot_prediction(eng.net)
ph.plot_variation(eng.net)

# save command to file
f = open("commands.txt", "a")
f.write(' '.join(sys.argv))
f.write('\n')
f.close()