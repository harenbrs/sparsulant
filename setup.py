import setuptools


setuptools.setup(
    name="sparsulant",
    version="0.1",
    author="Sebastian Harenbrock",
    author_email="harenbrs@tcd.ie",
    description="Sparse circulant matrices for SciPy",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/harenbrs/sparsulant",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.6"
)
