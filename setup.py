from setuptools import setup, find_packages

setup(
    name='surpRISe',
    version='0.1.0',
    author='Sergio Micó Rosa',
    author_email='sermiro@iteam.upv.es',
    description='surpRISe: Simulator Under Reconfigurable Propagation in RIS Environments',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sergiomico/surpRISe',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)