# Copyright 2019 Xiang Gao and collaborators.
# This program is distributed under the MIT license.
import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='TorchSnooper',
    author='Xiang Gao',
    author_email='qasdfgtyuiop@gmail.com',
    description="Debug PyTorch code using PySnooper.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/zasdfgbnm/TorchSnooper',
    packages=setuptools.find_packages(exclude=['tests']),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'pysnooper>=0.1.0',
        'numpy',
    ],
    tests_require=[
        'pytest',
        'torch',
        'python-toolbox',
        'coverage',
        'snoop',
    ],
)
