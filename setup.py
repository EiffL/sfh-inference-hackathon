from setuptools import find_packages
from setuptools import setup

setup(name='sfh',
      description='sfh package',
      author='AstroInfo',
      packages=find_packages(),
      install_requires=['astropy', 'tensorflow_datasets']
      )