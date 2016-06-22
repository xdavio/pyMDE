from setuptools import setup

setup(name='pymde',
      version='0.1',
      description='Python methods for Disparity Analysis',
      author='John Davis',
      author_email='johncliftondavis@gmail.com',
      url='',
      py_modules = ['utils', 'plot', 'est', 'gen', 'estPois', 'disparity'],
      install_requires=[
          'numpy', 'pandas'
      ],
      packages=['pymde']
     )
