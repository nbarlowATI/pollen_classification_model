#!/usr/bin/env python
from setuptools import find_packages, setup

requirements = []
with open("requirements.txt") as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

setup(
    name="pollen_classification_model",
    version="0.0.1",
    description="scivision plugin, using EfficientNetB3 model",
    ### TODO ###
    url="https://github.com/lizhiqihhh/pollen_classification_model",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
)
