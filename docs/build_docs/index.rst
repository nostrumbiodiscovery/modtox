.. modtox master file, created by
   sphinx-quickstart on Mon Dec  4 11:58:09 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

modtox
===========================================

We have concentrated our efforts on the High Risk-Off Target (HROT) set published by the Altman group at Stanford (Clin Transl Sci. 2016 Dec;9(6):311-320). These are 83 proteins that are predicted to bind low-dose drugs more frequently than high-dose drugs, i.e. proteins that are related to drugs which are administered at very low doses, because high doses lead to adverse events. Assessing whether small molecules interact with HROTs is useful in all phases of drug discovery. We are in the process of building a user-friendly platform to discriminate in a high-throughput manner whether compounds are toxic (binding to HROTs) or not.
The first step of the workflow consists in extracting 10 representative clusters from each molecular dynamics. Next, a curated dataset of active and decoy compounds are docked into these clusters. The docking scores are then used to train different machine learning models, so as to obtain a consistent and robust classifier that can predict whether a given compound hits an HROT target.
We have so far tested this approach on three systems: The androgen receptor (394 actives/404 decoys), CYP3A4 (168 actives/168 decoys), and the beta-2 adrenergic receptor (230 actives/230 decoys).


Github : https://github.com/danielSoler93/modtox


.. figure:: images/modtox.png
    :scale: 80%
    :align: center

Documentation
===================

.. toctree::
   installation/index.rst


.. toctree::
   tutorial/index.rst


.. toctree::
   changelog/index.rst

