#!/bin/bash

# Analyse MD
python -m modtox.cpptraj.analisis traj.xtc 198 --top init.top --RMSD --clust_type BS --last

# Ensemble docking from dude
python -m modtox.main traj.xtc 198 --dude adrb2 --top __init__.top --grid_mol 2 --dock  

#Ensemble docking active inactive 
python -m modtox.main traj.xtc 198 --actives active.sdf --inactives decoys.sdf --top __init__.top --grid_mol 3 --dock


