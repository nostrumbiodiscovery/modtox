Installation
====================

From Conda (recommended)
--------------------------

::

 conda create -c rdkit -c josan_bcn -c nostrumbiodiscovery -c ambermd -c conda-forge  -n modtox_conda python=3.6 modtox --yes

 source activate modtox_conda

 python -m modtox.main -h

From Source Code
---------------------

::

 git clone https://github.com/danielSoler93/modtox.git
 
 cd modtox

 python setup.py install

 conda install -c rdkit rdkit


From PyPi
-----------

::

  conda create -n modtox_pip python=3.6

  source activate modtox_pip  

  pip install modtox

  conda install -c rdkit rdkit
