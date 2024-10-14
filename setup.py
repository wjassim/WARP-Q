from setuptools import setup, find_packages
from os import path

working_directory = path.abspath(path.dirname(__file__))

with open(path.join(working_directory, "README_pypi.md"), encoding="utf-8") as f:
    long_description = f.read()

# with open("README.md", encoding="utf-8") as f:
#   long_description = f.read()


setup(
    name="warpq",  # The name of your package on PyPI
    version="1.5.2",  # Package version
    author="Wissam A Jassim",
    author_email="wissam.a.jassim@gmail.com",
    description="WARP-Q: Quality Prediction For Generative Neural Speech Codecs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wjassim/WARP-Q",
    packages=find_packages(),  # Automatically find and include all packages
    classifiers=[  # Meta information about the package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Minimum Python version requirement
    install_requires=["librosa", "pandas", "seaborn", "scipy", "tqdm", "pyvad", "numpy", "joblib", "matplotlib"],  # Dependencies
)
