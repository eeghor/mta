from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mta",
    version="0.0.8",
    description="Multi-Touch Attribution Models for Marketing Analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    url="https://github.com/eeghor/mta",
    author="Igor Korostil",
    author_email="eeghor@gmail.com",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
        "arrow>=1.0.0",
    ],
    package_data={"mta": ["data/*.csv.gz"]},
    keywords="attribution marketing multi-touch analytics markov shapley",
    project_urls={
        "Bug Reports": "https://github.com/eeghor/mta/issues",
        "Source": "https://github.com/eeghor/mta",
    },
)
