from setuptools import setup, find_packages

setup(
    name="dataecho",
    version="0.1.0",
    author="BD",
    author_email="",
    description="Survey and data analytics tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/brentlib/dataecho",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas"
    ],
)