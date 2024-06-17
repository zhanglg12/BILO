#!/usr/bin/env python
from datetime import datetime

from Engine import *

optobj = Options()
optobj.parse_args(*sys.argv[1:])


pid = os.getpid()
# save command to file
f = open("commands.txt", "a")
f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
f.write(f'  pid: {pid}')
f.write('\n')
f.write(' '.join(sys.argv))
f.write('\n')
f.close()


# set seed
set_seed(optobj.opts['seed'])

eng = Engine(optobj.opts)
eng.setup_problem()
eng.setup_trainer()

summary(eng.net)

eng.run()
# eng.trainer.save()

# eng.pde.dataset.to_device(eng.device)
eng.pde.visualize(savedir=eng.logger.get_dir())

