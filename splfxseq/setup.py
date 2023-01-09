from setuptools import setup
#from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import pysam
import numpy 
import glob
import os.path as op

setup(
    name="splfxseq",

    packages=['spliceprocess','splanl'],

    include_dirs = [numpy.get_include()]+pysam.get_include(),

    entry_points = {
        'console_scripts': [   
         'index_tagsorted_bam = spliceprocess.index_tagsorted_bam:main', 
       ]
    }
)


