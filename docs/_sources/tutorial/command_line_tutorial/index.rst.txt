From Command Line
==========================

Analyse moleculer dynamics
---------------------------------------


     Plot RMSD of trajectory, clusterize, extract last snapshot from MD trajectory

    ::

     python -m modtox.cpptraj.main traj*.(xtc, dcd, x, pdb...) resname --top topology.top --clust_type [all (allAtoms) , BS (BindinSite)] --rmsd --last


     python -m modtox.cpptraj.analisis modtox/tests/data/general/traj.pdb 198 --top modtox/tests/data/general/init.top --clust_type BS


Ensemble docking on molecular dynamics from dude
----------------------------------------------------------


     Perform clustering of the MD based on the binding site, followed by ensemble cross docking of a dude dataset

::

  python -m modtox.main taj_file resname --dude dude_folder --top topology_file --grid_mol number_of_the_molecule_to_be_used_as_grid (counted per TERS i.e. chainA/TER/chainB/TER/Ligand --> 3 ) --dock
  
  python -m modtox.main modtox/tests/data/general/traj.pdb 198 --dude modtox/tests/data/dude --top modtox/tests/data/general/init.top --grid_mol 2 --dock --clust_sieve 1


Perform ensemble docking from Active & Inactive sdf file
-------------------------------------------------------------

     Perform clustering of the MD based on the binding site, followed by ensemble cross docking of the active and inactive sdfs

::

  python -m modtox.main taj_file resname --actives active.sdf --inactives inactive.sdf --top topology_file --grid_mol number_of_the_molecule_to_be_used_as_grid (counted per TERS i.e. chainA/TER/chainB/TER/Ligand --> 3 ) --dock
  

  python -m modtox.main modtox/tests/data/general/traj.pdb 198 --active active.sdf --inactive decoys.sdf --top modtox/tests/data/general/init.top --grid_mol 3 --dock --clust_sieve 1


