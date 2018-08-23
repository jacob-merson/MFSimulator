from setuptools import setup, find_packages
from os.path import join,dirname


setup(

    name = 'MFSimulator',
    version='1.0.0',
    author="Jacob Merson",
    author_email="jacob.merson.17@ucl.ac.uk",
    python_requires='>=3',
    packages = find_packages(exclude=['*test']),
    install_requires = ['argparse','pytest','pyyaml','importlib','Mako','numpy','scipy',
    'pyopencl','matplotlib','scikit_learn'],
    py_modules = ['utils'],
    long_description=open(join(dirname(__file__),'README.md')).read()
    )



