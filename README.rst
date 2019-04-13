ModTox
================

Platform to check for toxicological effects based on
apo and holo simulations of the HROT dataset


Installation
=================

git clone https://github.com/danielSoler93/ModTox.git

cd ModTox

python setup.py install

python -m ModTox.main traj*.xtc resname --top topology.pdb


Analyse MDs
==================

To analayze MDs (RMSD, last snapshot & clusterization):

python -m ModTox.cpptraj.main traj*.(xtc, dcd, x, pdb...) resname --top topology.pdb --clust_type [all (allAtoms) , BS (BindinSite)] --rmsd --last

python -m ModTox.cpptraj.analisis 1nxk_0*.xtc 198 --top init.top --RMSD --clust_type BS --last


Perform Ensemble Docking over a trajectory with actives/decoys
================================================================

# Temporary 

Firs you need to harcode your Schrodinger Path under: ModTox/constants/constants.py (SCHR_PATH=) --> Temporary


# Perform ensemble docking from Dude:

python -m ModTox.main taj_file resname --dude dude_folder --top topology_file --grid_mol number_of_the_molecule_to_be_used_as_grid (counted per TERS i.e. chainA/TER/chainB/TER/Ligand --> 3 ) --dock

python -m ModTox.main 3d4s_holo.*.xtc TIM --dude adrb2 --top /data/ModTox/ldiaz_modtox_storage/HROT_3d4s/holo/3d4s_holo.top --grid_mol 3 --dock

# Perform ensemble docking from Active & Inactive sdf file:

python -m ModTox.main taj_file resname --actives active.sdf --inactives inactive.sdf --top topology_file --grid_mol number_of_the_molecule_to_be_used_as_grid (counted per TERS i.e. chainA/TER/chainB/TER/Ligand --> 3 ) --dock

python -m ModTox.main taj_file resname --actives active.sdf --inactives inactive.sdf --top topology_file --grid_mol 3 --dock

