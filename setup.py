from setuptools import setup, find_packages

setup(
    name='pulse_mode_analysis',  # Replace with your desired library name
    version='0.1.0',
    description='Analysis of waveforms',
    author='Martin Unland',  
    author_email='martin.e@unland.eu',  
    url='https://github.com/martinunland/pulse_mode_analysis',  
    packages=find_packages(),
    install_requires=[
        # List the packages required for your library here, e.g.
         'scipy',
         'numpy',
    ],
    python_requires='>=3.6',
)