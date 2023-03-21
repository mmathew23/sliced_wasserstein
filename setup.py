from setuptools import setup, find_packages


setup(
    name='sliced_wasserstein',
    version='0.1.0',
    author='Mathew J. Mathew',
    author_email='mathew.jude.mathew@gmail.com',
    description='A package to compute sliced wasserstein distance on pytorch tensors',
    long_description="""
    This package contains utils to help define your own sliced wasserstein distance for ML
    training. It currently supports Generalized Sliced Wasserstein distance and Max Generalized
    Sliced Wasserstein Distance. You can also add support for different projectors.
    """,
    long_description_content_type='text/markdown',
    url='https://github.com/mmathew23/sliced_wasserstein',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Environment :: GPU :: NVIDIA CUDA',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
        'torch',
    ],
)
