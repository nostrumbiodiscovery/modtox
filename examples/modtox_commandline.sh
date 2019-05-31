#!/bin/bash

# Analyse MD
python -m modtox.cpptraj.analisis ../tests/data/general/traj.pdb 198 --top ../tests/data/general/init.top --RMSD --clust_type BS --last

# Ensemble docking from dude
python -m modtox.main ../tests/data/general/traj.pdb 198 --dude ../tests/data/dude --top ../tests/data/general/init.top --grid_mol 2 --dock --clust_sieve 1 

#Ensemble docking active inactive 
python -m modtox.main ../tests/data/general/traj.pdb 198 --active ../tests/data/active_decoys/active.sdf --inactive ../tests/data/active_decoys/decoys.sdf --top ../tests/data/general/init.top --grid_mol 3 --dock --clust_sieve 1


