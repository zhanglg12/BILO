#!/bin/bash
# command for testing the results using piosson equation
common='smallrun False'
./runexp.py $common traintype basic   save_dir ./runs/testbasic     &
./runexp.py $common traintype init save_dir ./runs/testinit with_param True  &
wait    
./runexp.py $common traintype inverse restore ./runs/testinit save_dir ./runs/testinv with_param True  
wait    
