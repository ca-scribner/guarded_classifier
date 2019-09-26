import os
import sys

from setuptools import setup, find_packages

sys.path.append(os.path.abspath('./guarded_classifier'))
from version import VERSION

setup(name='guarded_classifier',
      version=VERSION,
      description='Meta-Classifier that guards against predicting classes with few training records',
      author='Andrew Scribner',
      install_requires=[
        'numpy>=1.14.0',
        'scikit-learn>=0.19.1',
      ],
      packages=find_packages(),
      python_requires='>3.6',
      )