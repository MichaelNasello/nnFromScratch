from setuptools import setup, find_packages

setup(
    name="Neural Networks from Scratch",
    version="1.0",
    author="Michael Nasello",
    install_requires=[
        "requests~=2.25.1",
        "matplotlib~=3.4.1",
        "Pillow~=8.2.0",
        "numpy~=1.19.5",
        "setuptools~=51.3.3",
    ],
    packages=find_packages()
)
