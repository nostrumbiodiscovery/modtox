from setuptools import find_packages, setup


setup(name='modtox',
      packages=find_packages(),
      version='1.0.0.3',
      license='MIT',
      description='Asses toxic effect of drugs based on MD simualtions of HROT dataset',
      author='Daniel Soler',
      author_email='daniel.soler@nostrumbiodiscovery.com',
      url='https://github.com/NostrumBioDiscovery/modtox',
      install_requires=['pytraj', 'mdtraj', 'matplotlib', 'xgboost', 'sklearn', 'prody', 'biopython', 'pandas', 'mordred', 'nltk', 'seaborn', 'umap-learn', 'tqdm', 'requests', 'chembl_webresource_client', 'pytest'],
     )
