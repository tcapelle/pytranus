#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree
import pip

from pip.req import parse_requirements
from setuptools import find_packages, setup, Command
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install
from setuptools.extension import Extension

# Package meta-data.
NAME = 'pytranus'
DESCRIPTION = 'Python Tranus (Lcal module) '
URL = 'https://gitlab.inria.fr/tcapelle/Tranus_Python'
EMAIL = 'thomascapelle@gmail.com'
AUTHOR = 'Thomas Capelle'

# What packages are required for this module to be executed?
install_reqs = parse_requirements('./requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]


class OverrideInstall(install):

    """
    Emulate sequential install of pip install -r requirements.txt
    To fix numpy bug in scipy, scikit in py2
    https://github.com/scikit-learn/scikit-learn/issues/4164
    """

    def run(self):
        for req in reqs:
            pip.main(["install", req])

# Extensions


class build_ext(_build_ext):

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your MANIFEST.in
# file!
with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)


class PublishCommand(Command):
    """Support setup.py publish."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except IOError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests',)),

    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    setup_requires=[
        'cython>=0.23.4',
        'numpy>=1.10.4',
        # setuptools 18.0 properly handles Cython extensions
        'setuptools>=18.0'
    ],
    install_requires=reqs,
    ext_modules=[
        Extension("pytranus.pylcal.utils.DX",
                  ["pytranus/pylcal/utils/DX.pyx"],
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp']
                  )
    ],
    include_package_data=True,
    license='ISC',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],

    cmdclass={
        'build_ext': build_ext,
        'install': OverrideInstall,
        'publish': PublishCommand
    },
)
