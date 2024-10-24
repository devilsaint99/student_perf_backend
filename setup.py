from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path,'r') as f:
        data = f.read()
        requirements = data.split('\n')
    for i,package in enumerate(requirements):
        package = package.strip()
        requirements[i] = package
    if '-e .' in requirements:
        requirements.remove('-e .')
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Anshul',
    author_email='anshul.suresh99@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
