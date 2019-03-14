from setuptools.command.test import test as TestCommand
from distutils.core import setup, Extension
import numpy.distutils.misc_util
import sys

if sys.version_info < (3,0):
	sys.exit('Sorry, Python 2 is not supported')

class ClassifierTest(TestCommand):

	def run_tests(self):
		import SNe_Early_Time_Classifier as setc
		errno = setc.test()
		sys.exit(errno)

AUTHOR = 'Tayler Quist, Ben Garza, David Jones'
AUTHOR_EMAIL = 'david.jones@ucsc.edu'
VERSION = '0.1dev'
LICENSE = 'BSD'
URL = 'sne-early-time-classifier.readthedocs.io'

setup(
	name='SNe_Early_Time_Classifier',
	version=VERSION,
	packages=['SNe_Early_Time_Classifier','SNe_Early_Time_Classifier.tests','SNe_Early_Time_Classifier.mangle'],
	cmdclass={'test': ClassifierTest},
	scripts=[],
	package_data={'': ['templates/Hsiao07.dat','templates/synphot_vega.dat','templates/flatnu.dat']},
	include_package_data=True,
	author=AUTHOR,
	author_email=AUTHOR_EMAIL,
	license=LICENSE,
	long_description=open('README.md').read(),
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
	install_requires=['numpy>=1.5.0',
					  'scipy>=0.9.0',
					  'astropy>=0.4.0',
					  'pysynphot>=0.9.12'],
	)
