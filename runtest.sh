#!/bin/bash
# command for testing the results using piosson equation
# common='smallrun False problem Poisson'
# ./runexp.py $common traintype basic   save_dir ./runs/testbasic 
# ./runexp.py $common traintype init save_dir ./runs/testinit with_param True
# ./runexp.py $common traintype inverse restore ./runs/testinit save_dir ./runs/testinv with_param True  


# need to run very long inverse
# common='smallrun False problem simpleode'
# ./runexp.py $common traintype basic   save_dir ./runs/ode_basic  datafile ./dataset/dataset_simple_exact.mat trainable_param a11
# ./runexp.py $common traintype init save_dir ./runs/ode_init2  datafile ./dataset/dataset_simple_init2.mat with_param True trainable_param a11
# ./runexp.py $common traintype inverse restore ./runs/ode_init2 save_dir ./runs/tmp with_param True  datafile ./dataset/dataset_simple_exact.mat trainable_param a11 



# ./runexp.py $common traintype basic   save_dir ./runs/ode_basic  datafile ./dataset/dataset_simple_exact.mat trainable_param a11,a22
# ./runexp.py $common traintype init save_dir ./runs/ode_init3  datafile ./dataset/dataset_simple_init3.mat with_param True trainable_param a11,a22
# ./runexp.py $common traintype inverse restore ./runs/ode_init3 save_dir ./runs/ode_init3_inv datafile ./dataset/dataset_simple_exact.mat with_param True trainable_param a11,a22


# ./runexp.py $common traintype inverse restore ./runs/ode_init3_inv save_dir ./runs/ode_init3_inv_cont datafile ./dataset/dataset_simple_exact.mat with_param True trainable_param a11,a22 optimizer lbfgs



# ./runexp.py $common traintype inverse restore ./runs/ode_init3 save_dir ./runs/ode_init3_inv_ datafile ./dataset/dataset_simple_exact.mat with_param True trainable_param a11,a22 burnin 10000


# ./runexp.py $common traintype inverse restore ./runs/ode_init3_inv_ save_dir ./runs/ode_init3_inv_2 datafile ./dataset/dataset_simple_exact.mat with_param True trainable_param a11,a22 burnin 10000


# common='smallrun False problem simpleode'
# # ./runexp.py $common traintype basic   save_dir ./runs/dampedosc_basic  datafile ./dataset/dataset_dampedosc_exact.mat trainable_param a11
# # ./runexp.py $common traintype init save_dir ./runs/dampedosc_init  datafile ./dataset/dataset_dampedosc_init_a11.mat with_param True trainable_param a11
# # ./runexp.py $common traintype init restore ./runs/dampedosc_init save_dir ./runs/dampedosc_init_lbfgs  datafile ./dataset/dataset_dampedosc_init_a11.mat with_param True trainable_param a11 optimizer lbfgs
# ./runexp.py $common traintype inverse restore ./runs/dampedosc_init save_dir ./runs/ode_init_inv datafile ./dataset/dataset_dampedosc_exact.mat with_param True trainable_param a11



# larger
# common='smallrun False problem simpleode width 128 burnin 50000'
# ./runexp.py $common traintype init save_dir ./runs/dampedosc_init_w128  datafile ./dataset/dataset_dampedosc_init_a11.mat with_param True trainable_param a11
# ./runexp.py $common traintype inverse restore ./runs/dampedosc_init_w128 save_dir ./runs/ode_init_w128_inv datafile ./dataset/dataset_dampedosc_exact.mat with_param True trainable_param a11

# smaller weight
common='smallrun False problem simpleode '
# ./runexp.py $common traintype init save_dir ./runs/dampedosc_init_resg-4  datafile ./dataset/dataset_dampedosc_init_a11.mat resgrad 1e-4 with_param True trainable_param a11 burnin 50000
# ./runexp.py $common traintype inverse restore ./runs/dampedosc_init_resg-4 save_dir ./runs/dampedosc_init_resg-4_inv datafile ./dataset/dataset_dampedosc_exact.mat resgrad 1e-4 with_param True trainable_param a11 burnin 10000



# common='smallrun False problem simpleode useFourierFeatures True'
# ./runexp.py $common traintype init save_dir ./runs/dampedosc_init_wff  datafile ./dataset/dataset_dampedosc_init_a11.mat with_param True trainable_param a11 burnin 50000
# ./runexp.py $common traintype inverse restore ./runs/dampedosc_init_wff save_dir ./runs/ode_init_wff_inv datafile ./dataset/dataset_dampedosc_exact.mat with_param True trainable_param a11


# triple step
common='smallrun False problem simpleode '
./runexp.py $common traintype init save_dir ./runs/dampedosc_init_0  datafile ./dataset/dataset_dampedosc_init_a11.mat resgrad None with_param True trainable_param a11 burnin 10000
./runexp.py $common traintype init restore ./runs/dampedosc_init_0  save_dir ./runs/dampedosc_init_1  datafile ./dataset/dataset_dampedosc_init_a11.mat resgrad 1e-3 with_param True trainable_param a11 burnin 10000
./runexp.py $common traintype inverse restore ./runs/dampedosc_init_1 save_dir ./runs/dampedosc_init_1_inv datafile ./dataset/dataset_dampedosc_exact.mat with_param True trainable_param a11