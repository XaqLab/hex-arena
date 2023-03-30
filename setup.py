from setuptools import setup, find_packages

with open('hexarena/VERSION.txt', 'r') as f:
    VERSION = f.readline().split('"')[1]

setup(
    name="hexarena",
    version=VERSION,
    author='Zhe Li',
    python_requires='>=3.9',
    packages=find_packages(),
    package_data={'hexarena': ['VERSION.txt']},
    install_requires=['irc>=0.3.1'],
)
