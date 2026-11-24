# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

from scverse_doc.registry import intersphinx

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = "sklearn-ann"
copyright = "2021, Frankie Robertson"
author = "Frankie Robertson"


# -- General configuration ---------------------------------------------------

extensions = [
    "scverse_doc",
    "sphinx_issues",
    "sphinx.ext.linkcode",
]
source_repository = "https://github.com/scikit-learn-contrib/sklearn-ann"
templates_path = ["_templates"]

intersphinx_mapping = intersphinx() | {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
xref_overrides = {
    "numpy._typing._array_like.ArrayLike": ("py:data", "numpy.typing.ArrayLike"),
    "numpy._typing._array_like.NDArray": ("py:data", "numpy.typing.NDArray"),
    **{
        f"numpy.{t}": ("py:attr", f"numpy.{t}") for t in ["float32", "float64", "int32"]
    },
}
# faiss’ docs have no Python API inventory
nitpick_ignore = [("py:class", "faiss.Index")]

autodoc_default_options = {
    "undoc-members": True,
}
autodoc_mock_imports = ["annoy", "faiss", "mlxvs", "pynndescent", "nmslib"]
