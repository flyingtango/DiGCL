from setuptools import setup, find_packages

setup(
    name='digcl',
    version='0.1.0',
    install_requires=['scikit-learn', 'pyyaml',
                      'pandas', 'torch-geometric==1.6.0'],
    packages=find_packages())
