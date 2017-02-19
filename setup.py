from setuptools import setup

VERSION=0.1

INSTALL_REQUIRES = [
    'scipy'
]

TEST_REQUIRES = [
    'pandas'
]

setup(
    name='rsk',
    version=0.1,
    packages=['rsk'],
    url='https://www.github.com/rwalk/rsk',
    author='Ryan Walker',
    author_email='ryan@ryanwalker.us',
    description='Repeated survey Kalman filter estimation routines.',
    license='MIT',
    test_requres=TEST_REQUIRES,
    install_requires=INSTALL_REQUIRES
)
