from setuptools import setup, find_packages

setup(
    name='ANI',
    version='0.1',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'torch',
        'numpy'
    ],
    python_requires='>=3.7',
)
