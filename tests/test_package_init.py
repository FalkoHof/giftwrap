import importlib.metadata

import numpy as np

import giftwrap
import giftwrap.analysis as analysis
from giftwrap import analysis_utils


def test_version_matches_installed_distribution_metadata():
    """`giftwrap.__version__` comes from the installed `giftwrap-sc` distribution."""
    assert giftwrap.__version__ == importlib.metadata.version("giftwrap-sc")


def test_numpy_compatibility_aliases_are_patched_on_import():
    """Importing giftwrap restores the removed `np.float_` and `np.infty` aliases."""
    assert np.float_ is np.float64
    assert np.infty == np.inf


def test_public_api_exports_reader_and_analysis_namespaces():
    """The package exposes `read_h5_file` and the `pp`/`pl`/`tl`/`sp` namespaces."""
    assert giftwrap.__all__ == ["read_h5_file", "pp", "pl", "tl", "sp"]
    assert giftwrap.read_h5_file is analysis_utils.read_h5_file
    assert giftwrap.pp is analysis.preprocess
    assert giftwrap.pl is analysis.plots
    assert giftwrap.tl is analysis.tools
    assert giftwrap.sp is analysis.spatial


def test_analysis_subpackage_exports_all_submodules():
    """`giftwrap.analysis` exports all four analysis submodules."""
    assert analysis.__all__ == ["preprocess", "plots", "tools", "spatial"]
    for name in analysis.__all__:
        assert hasattr(analysis, name)
