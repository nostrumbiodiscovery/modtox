From Command Line
==========================

Analyse moleculer dynamics
---------------------------------------


     Plot RMSD of trajectory, clusterize, extract last snapshot from MD trajectory

    ::

     python -m modtox.cpptraj.main traj*.(xtc, dcd, x, pdb...) resname --top topology.top --clust_type [all (allAtoms) , BS (BindinSite)] --rmsd --last


     python -m modtox.cpptraj.analisis 1nxk_0*.xtc 198 --top init.top --RMSD --clust_type BS --last


Ensemble docking on molecular dynamics from dude
----------------------------------------------------------


     Perform clustering of the MD based on the binding site, followed by ensemble cross docking of a dude dataset

    ::

  python -m modtox.main taj_file resname --dude dude_folder --top topology_file --grid_mol number_of_the_molecule_to_be_used_as_grid (counted per TERS i.e. chainA/TER/chainB/TER/Ligand --> 3 ) --dock
  
  python -m modtox.main 3d4s_holo.*.xtc TIM --dude adrb2 --top /data/modtox/ldiaz_modtox_storage/HROT_3d4s/holo/3d4s_holo.top --grid_mol 3 --dock


Perform ensemble docking from Active & Inactive sdf file
-------------------------------------------------------------

     Perform clustering of the MD based on the binding site, followed by ensemble cross docking of the active and inactive sdfs

    ::

  python -m modtox.main taj_file resname --actives active.sdf --inactives inactive.sdf --top topology_file --grid_mol number_of_the_molecule_to_be_used_as_grid (counted per TERS i.e. chainA/TER/chainB/TER/Ligand --> 3 ) --dock
  
  python -m modtox.main taj_file resname --actives active.sdf --inactives inactive.sdf --top topology_file --grid_mol 3 --dock


