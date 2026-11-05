"""Runs the real pipeline steps 1-5 on a tiny synthetic Flex library and checks the final counts."""

import numpy as np
import pandas as pd
import pytest

import giftwrap.step1_count_gapfills as step1
import giftwrap.step2_correct_umis as step2
import giftwrap.step3_correct_gapfill as step3
import giftwrap.step4_collect_counts as step4
import giftwrap.step5_summarize_counts as step5
import support
from giftwrap.analysis_utils import read_h5_file
from support import (
    CELLS,
    FLEX_BC1,
    FLEX_BC2,
    LHS_A,
    LHS_B,
    LHS_C,
    RHS_A,
    RHS_B,
    UMIS,
    flex_r1,
    flex_r2,
)

pytestmark = pytest.mark.integration
A, B = "GENEA c.1A>T", "GENEB c.2C>G"


def _library(probe_bc=None):
    def r2(lhs, gap, rhs):
        return flex_r2(lhs, gap, rhs, probe_bc)

    return (
        [(flex_r1(CELLS[0], UMIS[0]), r2(LHS_A, "TTG", RHS_A))] * 3
        + [(flex_r1(CELLS[0], support.mutate(UMIS[0], 6)), r2(LHS_A, "TTG", RHS_A))]
        + [(flex_r1(CELLS[0], UMIS[1]), r2(LHS_B, "GCAT", RHS_B))] * 2
        + [(flex_r1(CELLS[0], UMIS[1]), r2(LHS_B, "GCTT", RHS_B))]
        + [
            (flex_r1(CELLS[1], UMIS[2]), r2(LHS_A, "TAG", RHS_A)),
            (flex_r1(CELLS[1], UMIS[3]), r2(LHS_B, "GCAT", RHS_B)),
            (flex_r1(CELLS[1], UMIS[4]), r2(LHS_C, "TTG", RHS_A)),
            ("N" * 16 + UMIS[5], r2(LHS_A, "TTG", RHS_A)),
        ]
    )


def _exit_ok(func, *args):
    with pytest.raises(SystemExit) as exc:
        func(*args)
    assert exc.value.code == 0


def _run_pipeline(tmp_path, whitelist, pairs, cores=1, multiplex=1):
    probes = support.write_probe_file(tmp_path / "probes.tsv")
    r1, r2 = support.write_fastq_pair(tmp_path / "fastq", pairs)
    out = tmp_path / "out"
    _exit_ok(
        step1.run,
        str(probes),
        -1,
        str(r1),
        str(r2),
        None,
        str(out),
        cores,
        4,
        1,
        "Flex",
        None,
        False,
        multiplex,
        None,
        None,
        None,
        False,
        False,
        False,
        None,
        [str(whitelist)],
        False,
        -1,
        False,
    )
    _exit_ok(step2.run, str(out), 1, cores, 10, False)
    _exit_ok(step3.run, str(out), cores, 10)
    _exit_ok(step4.run, str(out), cores, False, multiplex > 1, False, 10)
    step5.run(str(out), False, None, False, 0)
    return out


def test_singleplex_library_produces_expected_counts_and_reports(
    tmp_path, whitelist, no_cellranger
):
    """A small singleplex library yields the expected reads, counts, PCR layers and reports."""
    out = _run_pipeline(tmp_path, whitelist, _library())

    metrics = (
        pd.read_table(out / "fastq_metrics.tsv").set_index("metric")["value"].to_dict()
    )
    assert metrics["TOTAL_READS"] == 11 and metrics["PROBE_CONTAINING_READS"] == 9
    assert metrics["FILTERED_NO_LHS"] == 1 and metrics["FILTERED_NO_CELL_BARCODE"] == 1

    reads = [
        line.split("\t") for line in support.read_lines(out / "probe_reads.tsv.gz")
    ]
    assert reads[0] == [
        "cell_idx",
        "probe_idx",
        "probe_barcode",
        "umi",
        "gapfill",
        "pcr_duplicate_count",
        "percent_supporting",
    ]
    assert sorted((r[0], r[1], r[3], r[4], r[5]) for r in reads[1:]) == [
        ("0", "0", UMIS[0], "TTG", "4"),
        ("0", "1", UMIS[1], "GCAT", "3"),
        ("1", "0", UMIS[2], "TAG", "1"),
        ("1", "1", UMIS[3], "GCAT", "1"),
    ]

    adata = read_h5_file(out / "counts.1.h5")
    assert adata.obs_names.tolist() == [f"{CELLS[0]}-1", f"{CELLS[1]}-1"]
    assert adata.var_names.tolist() == [f"{A}|TAG", f"{A}|TTG", f"{B}|GCAT"]
    np.testing.assert_array_equal(adata.X.toarray(), [[0, 1, 1], [1, 0, 1]])
    np.testing.assert_array_equal(
        adata.layers["total_reads"].toarray(), [[0, 4, 3], [1, 0, 1]]
    )
    np.testing.assert_array_equal(
        adata.layers["X_pcr_threshold_2"].toarray(), [[0, 1, 1], [0, 0, 0]]
    )
    np.testing.assert_array_equal(
        adata.layers["X_pcr_threshold_3"].toarray(), [[0, 1, 1], [0, 0, 0]]
    )
    assert adata.var.loc[f"{A}|TTG", "expected_gapfill"] == "TTG"
    assert adata.var.loc[f"{A}|TTG", "reference_gapfill"] == "TAG"

    summary = pd.read_table(out / "counts.1.summary.tsv").set_index("statistic")[
        "value"
    ]
    assert summary["TOTAL_CELLS"] == 2 and summary["TOTAL_UMIS"] == 4
    assert summary["SEQUENCING_SATURATION"] == pytest.approx(1 - 4 / 9)
    assert (out / "counts.1.summary.html").stat().st_size > 0
    assert (out / "counts.1.summary.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.multiprocess
def test_multiplexed_library_is_split_by_probe_barcode(
    tmp_path, whitelist, no_cellranger
):
    """A two-plex library yields one counts file per probe barcode with that plex's cells."""
    pairs = _library(FLEX_BC1)[:7] + _library(FLEX_BC2)[7:9]
    out = _run_pipeline(tmp_path, whitelist, pairs, cores=2, multiplex=2)

    plex1 = read_h5_file(out / "counts.1.h5")
    plex2 = read_h5_file(out / "counts.2.h5")
    assert plex1.obs_names.tolist() == [f"{CELLS[0]}{FLEX_BC1}-1"]
    assert plex2.obs_names.tolist() == [f"{CELLS[1]}{FLEX_BC2}-1"]
    np.testing.assert_array_equal(plex1.X.toarray(), [[1, 1]])
    np.testing.assert_array_equal(plex2.X.toarray(), [[1, 1]])
    assert plex1.var_names.tolist() == [f"{A}|TTG", f"{B}|GCAT"]
    assert plex2.var_names.tolist() == [f"{A}|TAG", f"{B}|GCAT"]
    for plex in ("1", "2"):
        assert (out / f"counts.{plex}.summary.tsv").exists()
