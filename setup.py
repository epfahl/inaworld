from setuptools import setup, find_packages

setup(
    name='inaworld',
    version='0.1',
    description='Give a movie description, get tags',
    url='https://github.com/epfahl/inaworld',
    author='Eric Pfahl',
    packages=find_packages(),
    install_requires=[
        'toolz',
        'nltk'])
