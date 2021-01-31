from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize("*.pyx", compiler_directives={"language_level": "3"})
)
