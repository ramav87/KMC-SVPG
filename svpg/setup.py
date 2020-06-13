from setuptools import setup, find_packages



setup(
    name='svpg',
    version='0.01',
    packages=['svpg'],
    install_requires=['tensorflow', 'scipy', 'numpy', 'gym'],
    url='https://code.ornl.gov/ai/rl/svpg',
    license='MIT',
    include_package_data=True

)
