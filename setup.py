from setuptools import setup, find_packages

setup(
    name="superglue",
    version="1.0",
    author="magic leap",
    description="SuperGlue Inference and Evaluation Demo Script",
    url="https://github.com/magicleap/SuperGluePretrainedNetwork", 
    package_data={'superglue': ['models/weights/*.pth'],},
    packages=find_packages()
)