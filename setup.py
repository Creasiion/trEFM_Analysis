from setuptools import setup, find_packages

from os import path

setup(
    name='trEFM Analysis UW',
    version='0.0.1',
    description='trEFM Simulation, data uploader, and calibration curve',
    author='Imani Cage',
    author_email='imani.l.cage@gmail.com',
    license='MIT',
    url='https://github.com/Creasiion/trEFM_Analysis',

    packages=find_packages(),

    install_requires=['numpy>=1.23.5',
                      'scipy>=1.10.1'
                      ]


)
