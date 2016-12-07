from setuptools import setup

VERSION=0.1
INSTALL_REQUIRES = [
    'scipy'
]

setup(
    name='rsk',
    version=0.1,
    url='https://www.github.com/rwalk/rsk',
    author='Ryan Walker',
    author_email='ryan@ryanwalker.us',
    description='Repeated survey Kalman filter estimation routines.',
    license='MIT',
    install_requires=INSTALL_REQUIRES
)
