"""Python setup.py for ritm_annotation package"""
import io
import os
from setuptools import find_packages, setup
from Cython.Build import cythonize
from distutils.core import Extension


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("ritm_annotation", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


def ext_modules():
    import numpy as np
    includes = []
    libraries = []

    includes.append(np.get_include())
    if os.name == "posix":
        libraries.append("m")
    modules = []
    modules += cythonize(Extension(
        "*",
        ["ritm_annotation/**/*.pyx"],
        include_dirs=includes,
        libraries=libraries,
    ))
    return modules

setup(
    name="ritm_annotation",
    version=read("ritm_annotation", "VERSION"),
    description="Awesome ritm_annotation created by lucasew",
    url="https://github.com/lucasew/ritm_annotation/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="lucasew",
    packages=find_packages(exclude=["tests", ".github"]),
    ext_modules=ext_modules(),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["ritm_annotation = ritm_annotation.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
