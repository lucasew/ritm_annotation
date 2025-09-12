from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from pathlib import Path

extensions = [
    Extension(
        "ritm_annotation.utils.cython._get_dist_maps",
        sources=["ritm_annotation/utils/cython/_get_dist_maps.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="ritm_annotation",
    version=Path("./ritm_annotation/VERSION").read_text().strip(),
    ext_modules = cythonize(extensions),
    py_modules=["ritm_annotation"]
)
