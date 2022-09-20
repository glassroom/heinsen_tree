# coding: utf-8
from setuptools import setup

setup(name='heinsen_tree',
    version='1.0',
    description='Reference implementation of "Tree Methods for Hierarchical Classification in Parallel" (Heinsen, 2022).',
    url='https://github.com/glassroom/heinsen_tree',
    author='Franz A. Heinsen',
    author_email='franz@glassroom.com',
    license='MIT',
    packages=['glassroom'],
    requires_python='>=3.6.0',
    install_requires='torch',
    zip_safe=False)
