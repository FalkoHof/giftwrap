import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import support  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic_and_clean():
    np.random.seed(0)
    yield
    plt.close("all")


@pytest.fixture
def whitelist(tmp_path) -> Path:
    (tmp_path / "wl").mkdir()
    return support.write_whitelist(tmp_path / "wl" / "barcodes.tsv.gz", support.CELLS)


@pytest.fixture
def flex_tech(whitelist):
    from giftwrap.pipeline_utils import FlexFormatInfo

    return FlexFormatInfo(barcode_list=[whitelist])


@pytest.fixture
def flex_v2_tech(whitelist):
    from giftwrap.pipeline_utils import FlexV2FormatInfo

    return FlexV2FormatInfo(barcode_list=[whitelist])


@pytest.fixture
def visium_tech():
    from giftwrap.pipeline_utils import VisiumFormatInfo

    return VisiumFormatInfo(
        version=1,
        barcode_dir=Path(__file__).parents[1] / "src" / "giftwrap" / "resources",
    )


@pytest.fixture
def no_cellranger(monkeypatch):
    real_which = shutil.which

    def which(name, *args, **kwargs):
        if name in ("cellranger", "spaceranger"):
            return None
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", which)


HD_BC1 = ["ACGTACGTACGTAC", "TTGCAAGCTTGCAA", "GACTGACTGACTGA"]
HD_BC2 = ["CATGCATGCATGCA", "AGTCCAGTTGGACA"]


@pytest.fixture
def fake_spaceranger(tmp_path, monkeypatch):
    """A minimal spaceranger install tree whose slide-design protobuf yields tiny bc1/bc2 oligo sets."""
    root = tmp_path / "spaceranger-x"
    (root / "bin").mkdir(parents=True)
    exe = root / "bin" / "spaceranger"
    exe.write_text("#!/bin/sh\n")
    spatial = root / "lib" / "python" / "cellranger" / "spatial"
    spatial.mkdir(parents=True)
    (spatial / "visium_hd_schema_pb2.py").write_text(textwrap.dedent(f"""
        class _TwoPart:
            bc1_oligos = {HD_BC1!r}
            bc2_oligos = {HD_BC2!r}

        class VisiumHdSlideDesign:
            def __init__(self):
                self.two_part = _TwoPart()

            def ParseFromString(self, data):
                return len(data)
    """))
    barcode_dir = tmp_path / "hd_resources"
    barcode_dir.mkdir()
    (barcode_dir / "visium_hd_v1.slide").write_bytes(b"\x00")
    shutil.copy(
        Path(__file__).parents[1]
        / "src"
        / "giftwrap"
        / "resources"
        / "chemistry_defs.json",
        barcode_dir / "chemistry_defs.json",
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "visium_hd_schema_pb2", raising=False)
    return {
        "exe": exe,
        "barcode_dir": barcode_dir,
        "root": root,
        "bc1": HD_BC1,
        "bc2": HD_BC2,
    }
