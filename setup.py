# Copyright 2019 Xiang Gao and collaborators.
# This program is distributed under the MIT license.
import setuptools


setuptools.setup(
    name='TorchSnooper',
    author='Xiang Gao',
    author_email='qasdfgtyuiop@gmail.com',
    description="A poor man's debugger for PyTorch.",
    url='https://github.com/zasdfgbnm/TorchSnooper',
    packages=setuptools.find_packages(exclude=['tests']),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'pysnooper',
    ],
    extras_require={
        'tests': {
            'pytest',
            'python-toolbox',
        },
    },
)