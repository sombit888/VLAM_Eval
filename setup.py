from setuptools import setup, find_packages

setup(
    name="VLAM_Eval",               # Name of the package
    version="0.1.0",                      # Initial version
    description="A tool for evaluation metrics and utilities.",
    author="Sombit",
    author_email="sombit.dey@insait.ai",
    url="https://github.com/sombit888/VLAM_Eval",  # Replace with your GitHub repo URL
    packages=find_packages(),             # Automatically find all packages
    install_requires=[],                  # Add dependencies here or use requirements.txt
    python_requires=">=3.10",              # Specify Python version compatibility
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

