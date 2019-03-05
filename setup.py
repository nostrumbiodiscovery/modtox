from distutils.core import setup


setup(name='ModTox Platform',
      version='1.0.0',
      description='Asses toxic effect of drugs based on MD simualtions of HROT dataset',
      author='Daniel Soler',
      author_email='daniel.soler@e-campus.uab.cat',
      url='https://www.python.org/sigs/distutils-sig/',
      install_requires=['pytraj', 'mdtraj'],
     )
