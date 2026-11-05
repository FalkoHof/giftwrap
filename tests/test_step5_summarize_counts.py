import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sankeyflow import Sankey

import giftwrap
import giftwrap.step5_summarize_counts as step5
import support
from giftwrap.analysis_utils import read_h5_file
from giftwrap.pipeline_utils import write_manifest
from giftwrap.step5_summarize_counts import (
    best_fit,
    density,
    make_pcr_duplicate_histogram,
    make_pdf_report,
    make_sankey,
    summarize_counts,
)

CELLS = [f"{c}-1" for c in support.CELLS[:4]]
PROBES = [
    ("GENEA c.1A>T", "TTG"),
    ("GENEA c.1A>T", "TAG"),
    ("GENEB c.2C>G", "GCAT"),
    ("GENEC 0bp", ""),
]
X = np.array([[2, 1, 0, 1], [0, 0, 3, 1], [1, 0, 0, 0], [0, 0, 0, 0]])
STEP3_HEADER = [
    "cell_idx",
    "probe_idx",
    "probe_barcode",
    "umi",
    "gapfill",
    "pcr_duplicate_count",
    "percent_supporting",
]
FLAT_HEADER = [
    "cell_barcode",
    "lhs_probe",
    "rhs_probe",
    "called_probe",
    "gapfill",
    "pcr_duplicates",
    "percent_supporting",
    "umi",
]
FASTQ_METRICS = {
    "TOTAL_READS": 100,
    "PROBE_CONTAINING_READS": 60,
    "POSSIBLE_PROBES": 3,
    "PROBES_ENCOUNTERED": 3,
    "EXACT": 50,
    "CORRECTED_LHS": 4,
    "CORRECTED_RHS": 3,
    "CORRECTED_BARCODE": 3,
    "FILTERED_NO_LHS": 20,
    "FILTERED_NO_RHS": 10,
    "FILTERED_NO_CELL_BARCODE": 5,
    "FILTERED_NO_CONSTANT": 3,
    "FILTERED_NO_PROBE_BARCODE": 2,
    "R1_CYCLES": 28,
    "R2_CYCLES": 90,
}


def _probe_metadata():
    return pd.DataFrame(
        {
            "name": ["GENEA c.1A>T", "GENEB c.2C>G", "GENEC 0bp"],
            "gene": ["GENEA", "GENEB", "GENEC"],
            "lhs_probe": [support.LHS_A, support.LHS_B, support.LHS_C],
            "rhs_probe": [support.RHS_A, support.RHS_B, support.RHS_A],
            "gap_probe_sequence": ["TTG", "GCAT", "NA"],
            "original_gap_probe_sequence": ["TAG", "GCCT", "NA"],
            "index": [0, 1, 2],
        }
    )


@pytest.fixture
def run_dir(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    support.write_counts_h5(
        out / "counts.1.h5",
        X,
        PROBES,
        CELLS,
        total_reads=X * 2,
        cell_index=[0, 1, 2, 3],
        probe_index=[0, 0, 1, 2],
        probe_metadata=_probe_metadata(),
    )
    pd.DataFrame(
        {"metric": list(FASTQ_METRICS), "value": list(FASTQ_METRICS.values())}
    ).to_csv(out / "fastq_metrics.tsv", sep="\t", index=False)
    write_manifest(
        out,
        support.manifest_df(include_control=True),
        {"technology": "Flex", "multiplexed": False},
    )
    support.write_tsv(
        out / "probe_reads.tsv.gz",
        STEP3_HEADER,
        [
            [0, 0, 1, "U1", "TTG", 2, 1.0],
            [0, 0, 1, "U2", "TTG", 3, 1.0],
            [0, 0, 1, "U3", "TAG", 2, 1.0],
            [0, 2, 1, "U4", "", 2, 1.0],
            [1, 1, 1, "U5", "GCAT", 1, 1.0],
            [1, 1, 1, "U6", "GCAT", 2, 1.0],
            [1, 1, 1, "U7", "GCAT", 3, 1.0],
            [1, 2, 1, "U8", "", 2, 1.0],
            [2, 0, 1, "U9", "TTG", 2, 1.0],
        ],
    )
    support.write_tsv(
        out / "flat_counts.1.tsv.gz",
        FLAT_HEADER,
        [
            [CELLS[0], "GENEA c.1A>T", "GENEA c.1A>T", 1, "TTG", 1, 1.0, "U1"],
            [CELLS[0], "GENEA c.1A>T", "GENEA c.1A>T", 1, "TTG", 3, 1.0, "U2"],
            [
                f"{support.CELLS[1]}{support.FLEX_BC1}-1",
                "GENEB c.2C>G",
                "GENEB c.2C>G",
                1,
                "GCAT",
                2,
                1.0,
                "U6",
            ],
            [CELLS[3], "GENEA c.1A>T", "GENEA c.1A>T", 1, "TTG", 5, 1.0, "U9"],
        ],
    )
    return out


def _wta(directory, barcodes, genes):
    support.write_10x_mtx(directory / "filtered_feature_bc_matrix", barcodes, genes)
    return directory


def _summarize(out, cellranger=None, flatten=False, reads_per_gapfill=0):
    paths = {
        k: out / f"counts.1.{k}"
        for k in ("summary.tsv", "summary.pdf", "summary.html", "filtered.h5")
    }
    paths["flat"] = out / "flat_counts.1.filtered.tsv.gz"
    summarize_counts(
        out / "counts.1.h5",
        paths["summary.tsv"],
        paths["summary.pdf"],
        paths["summary.html"],
        paths["filtered.h5"],
        paths["flat"],
        cellranger,
        flatten,
        reads_per_gapfill,
        "1",
    )
    return paths


def _stats(path):
    return pd.read_table(path).set_index("statistic")["value"].to_dict()


def _pdf_pages(path):
    return len(re.findall(rb"/Type\s*/Page[^s]", Path(path).read_bytes()))


class TestHelpers:
    def test_density_returns_positive_estimate_per_point(self):
        """The kernel density estimate returns one positive value per point."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 1.5])
        y = np.array([1.0, 0.5, 2.0, 2.5, 1.0])
        d = density(x, y)
        assert d.shape == (5,) and (d > 0).all()

    def test_best_fit_evaluates_linear_fit_at_unique_x(self):
        """The least-squares line is evaluated once per unique x value."""
        np.testing.assert_allclose(
            best_fit(np.array([0, 1, 2, 2]), np.array([1, 3, 5, 5])), [1, 3, 5]
        )

    def test_sankey_is_built_from_metrics_and_draws(self):
        """Read and count metrics build a Sankey diagram that draws on an axis."""
        sankey = make_sankey(FASTQ_METRICS, {"TOTAL_UMIS": 9, "TOTAL_CELLS": 4})
        assert isinstance(sankey, Sankey)
        fig, ax = plt.subplots()
        sankey.draw(ax)
        assert ax.patches or ax.collections

    def test_sankey_treats_missing_metrics_as_zero(self):
        """Metrics missing from the input count as zero."""
        sankey = make_sankey(
            {"TOTAL_READS": 10, "PROBE_CONTAINING_READS": 6, "EXACT": 6},
            {"TOTAL_UMIS": 4},
        )
        assert isinstance(sankey, Sankey)

    def test_pcr_histogram_uses_log_scale_and_marks_cutoff(self, tmp_path):
        """Duplicates are binned as log10(n + 1), with a cutoff line only when a cutoff is set."""
        reads = support.write_tsv(
            tmp_path / "r.tsv.gz",
            STEP3_HEADER,
            [[0, 0, 1, "U", "A", 1, 1.0], [0, 0, 1, "U", "A", 9, 1.0], ["bad"]],
        )
        fig, ax = plt.subplots()
        make_pcr_duplicate_histogram(reads, 0, ax)
        assert len(ax.patches) == 100 and not ax.lines
        assert ax.get_xlabel() == "Log10(PCR Duplicates + 1)"
        fig, ax = plt.subplots()
        make_pcr_duplicate_histogram(reads, 3, ax)
        assert ax.lines[0].get_xdata()[0] == pytest.approx(np.log10(4))


class TestMakePdfReport:
    def _inputs(self, run_dir):
        pd.DataFrame(
            {"statistic": ["TOTAL_UMIS", "TOTAL_CELLS"], "value": [9, 4]}
        ).to_csv(run_dir / "counts.1.summary.tsv", sep="\t", index=False)
        return run_dir / "counts.1.summary.pdf", read_h5_file(run_dir / "counts.1.h5")

    def test_report_without_wta_has_seven_pages(self, run_dir):
        """Without WTA data, the PDF report has seven pages."""
        pdf, gapfill = self._inputs(run_dir)
        make_pdf_report(pdf, gapfill, None, run_dir / "probe_reads.tsv.gz", 2)
        assert pdf.read_bytes().startswith(b"%PDF") and _pdf_pages(pdf) == 7

    def test_report_with_matching_wta_adds_correlation_page(self, run_dir, tmp_path):
        """Shared WTA genes add a correlation page; missing gene names are guessed from probes."""
        wta = step5.read_wta(_wta(tmp_path / "wta", CELLS, ["GENEA", "GENEB", "OTHER"]))
        pdf, gapfill = self._inputs(run_dir)
        del gapfill.var["gene"]
        make_pdf_report(pdf, gapfill, wta, run_dir / "probe_reads.tsv.gz", 0)
        assert _pdf_pages(pdf) == 8
        assert gapfill.var["gene"].tolist() == ["GENEA", "GENEA", "GENEB", "GENEC"]

    def test_report_skips_wta_page_without_shared_genes(
        self, run_dir, tmp_path, capsys
    ):
        """WTA data without shared genes adds no page and prints a warning."""
        wta = step5.read_wta(_wta(tmp_path / "wta", CELLS, ["X1", "X2"]))
        pdf, gapfill = self._inputs(run_dir)
        make_pdf_report(pdf, gapfill, wta, run_dir / "probe_reads.tsv.gz", 0)
        assert "No gene names match" in capsys.readouterr().out
        assert _pdf_pages(pdf) == 7

    def test_missing_matplotlib_backend_aborts_gracefully(
        self, run_dir, monkeypatch, capsys
    ):
        """Without matplotlib's PDF backend, no report is written and a message is printed."""
        monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_pdf", None)
        pdf, gapfill = self._inputs(run_dir)
        make_pdf_report(pdf, gapfill, None, run_dir / "probe_reads.tsv.gz", 0)
        assert "Matplotlib not found" in capsys.readouterr().out
        assert not pdf.exists()


class TestSummarizeCounts:
    def test_writes_statistics_and_reports(self, run_dir):
        """Without filters, exact statistics and the PDF, HTML and metrics CSV reports are written."""
        paths = _summarize(run_dir)
        stats = _stats(paths["summary.tsv"])
        expected = {
            "TOTAL_CELLS": 4,
            "TOTAL_UMIS": 9,
            "GAPFILL_CONTAINING_CELLS": 3,
            "UMIS_PER_CELL_MEAN": 2.25,
            "UMIS_PER_CELL_MEDIAN": 2.5,
            "UMIS_PER_CELL_STD": np.std([4, 4, 1, 0]),
            "UMIS_PER_CELL_MIN": 0,
            "UMIS_PER_CELL_MIN_EXCLUDING_ZERO": 1,
            "UMIS_PER_CELL_MAX": 4,
            "CELLS_PER_GAPFILL_MEAN": 1.5,
            "CELLS_PER_GAPFILL_MEDIAN": 1.5,
            "CELLS_PER_GAPFILL_STD": 0.5,
            "CELLS_PER_GAPFILL_MIN": 1,
            "CELLS_PER_GAPFILL_MAX": 2,
            "SEQUENCING_SATURATION": 2 / 3,
        }
        assert list(stats) == list(expected)
        for key, value in expected.items():
            assert stats[key] == pytest.approx(value), key
        assert _pdf_pages(paths["summary.pdf"]) == 7
        assert "<html" in paths["summary.html"].read_text().lower()
        assert (run_dir / "counts.1.metrics.csv").exists()
        assert not paths["filtered.h5"].exists()

    def test_cellranger_output_filters_and_pads_cells(self, run_dir, tmp_path, capsys):
        """CellRanger cells filter and pad the counts before they are summarized."""
        wta = _wta(
            tmp_path / "wta",
            [CELLS[0], CELLS[1], f"{support.CELLS[5]}-1"],
            ["GENEA", "GENEB"],
        )
        paths = _summarize(run_dir, cellranger=wta)
        assert "Cellranger identified 3 cells" in capsys.readouterr().out
        filtered = read_h5_file(paths["filtered.h5"])
        assert filtered.obs_names.tolist() == [
            CELLS[0],
            CELLS[1],
            f"{support.CELLS[5]}-1",
        ]
        assert _stats(paths["summary.tsv"])["TOTAL_CELLS"] == 3
        assert _pdf_pages(paths["summary.pdf"]) == 8

    def test_barcode_only_cellranger_output_filters_cells(
        self, run_dir, monkeypatch, capsys
    ):
        """When only barcodes can be read from CellRanger output, they still filter the counts."""
        monkeypatch.setattr(
            step5, "read_wta", lambda *a, **k: np.array([CELLS[1], CELLS[2]])
        )
        paths = _summarize(run_dir, cellranger=Path("wta"))
        assert "partially parsed" in capsys.readouterr().out
        assert _stats(paths["summary.tsv"])["TOTAL_CELLS"] == 2

    def test_unparseable_cellranger_output_is_ignored(
        self, run_dir, monkeypatch, capsys
    ):
        """Unreadable CellRanger output is ignored with a message."""
        monkeypatch.setattr(step5, "read_wta", lambda *a, **k: None)
        paths = _summarize(run_dir, cellranger=Path("wta"))
        assert "unable to parse" in capsys.readouterr().out
        assert _stats(paths["summary.tsv"])["TOTAL_CELLS"] == 4

    def test_minimum_reads_per_gapfill_filters_low_support_umis(self, run_dir):
        """`reads_per_gapfill` removes UMIs with fewer reads before summarizing."""
        paths = _summarize(run_dir, reads_per_gapfill=2)
        np.testing.assert_array_equal(
            read_h5_file(paths["filtered.h5"]).X.toarray(),
            [[2, 1, 0, 1], [0, 0, 2, 1], [1, 0, 0, 0], [0, 0, 0, 0]],
        )
        assert _stats(paths["summary.tsv"])["TOTAL_UMIS"] == 8

    def test_flattened_counts_are_filtered_to_cellranger_cells(self, run_dir, tmp_path):
        """The flat file keeps its header and rows of CellRanger cells, including plex-sequence barcodes."""
        wta = _wta(tmp_path / "wta", [CELLS[0], CELLS[1]], ["GENEA"])
        paths = _summarize(run_dir, cellranger=wta, flatten=True)
        rows = [line.split("\t") for line in support.read_lines(paths["flat"])]
        assert rows[0] == FLAT_HEADER
        assert [(r[0], r[7]) for r in rows[1:]] == [
            (CELLS[0], "U1"),
            (CELLS[0], "U2"),
            (f"{support.CELLS[1]}{support.FLEX_BC1}-1", "U6"),
        ]

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=ValueError,
        reason="the PCR-duplicate filter casts the flat-file header to int",
    )
    def test_flattened_counts_are_filtered_by_reads_per_gapfill(self, run_dir):
        """The flat file keeps its header and the UMIs with at least `reads_per_gapfill` reads."""
        paths = _summarize(run_dir, flatten=True, reads_per_gapfill=2)
        rows = support.read_lines(paths["flat"])
        assert rows[0].split("\t") == FLAT_HEADER
        assert [r.split("\t")[7] for r in rows[1:]] == ["U2", "U6", "U9"]

    def test_pdf_failures_do_not_prevent_the_html_report(
        self, run_dir, monkeypatch, capsys
    ):
        """A failing PDF report is reported, but the HTML report is still written."""

        def boom(*args):
            raise RuntimeError("pdf")

        monkeypatch.setattr(step5, "make_pdf_report", boom)
        paths = _summarize(run_dir)
        assert "Failed to generate PDF report." in capsys.readouterr().out
        assert paths["summary.html"].exists()


class TestRun:
    @pytest.fixture
    def recorded(self, monkeypatch):
        calls = []
        monkeypatch.setattr(step5, "summarize_counts", lambda *a: calls.append(a))
        return calls

    def _touch_counts(self, out, *names):
        out.mkdir(exist_ok=True)
        for name in names:
            (out / name).write_bytes(b"")
        return out

    def test_counts_files_are_summarized_in_plex_order_with_matching_cellranger(
        self, tmp_path, recorded, capsys
    ):
        """Counts files are summarized in numeric plex order, each with its own CellRanger output and paths."""
        out = self._touch_counts(
            tmp_path / "o",
            "counts.10.h5",
            "counts.2.h5",
            "counts.2.filtered.h5",
            "other.h5",
        )
        step5.run(str(out), True, ["wta2", "wta10"], True, 3)
        assert [c[0].name for c in recorded] == ["counts.2.h5", "counts.10.h5"]
        first = recorded[0]
        assert first[1:6] == (
            out / "counts.2.summary.tsv",
            out / "counts.2.summary.pdf",
            out / "counts.2.summary.html",
            out / "counts.2.filtered.h5",
            out / "flat_counts.2.filtered.tsv.gz",
        )
        assert first[6:] == (Path("wta2"), True, 3, "2")
        assert recorded[1][6] == Path("wta10") and recorded[1][9] == "10"
        assert "WTA CellRanger output provided." in capsys.readouterr().out

    def test_single_cellranger_string_and_count_mismatch_warning(
        self, tmp_path, recorded, capsys
    ):
        """A single CellRanger path applies to the first plex only, with a mismatch warning."""
        out = self._touch_counts(tmp_path / "o", "counts.1.h5", "counts.2.h5")
        step5.run(str(out), False, "wta", False, 0)
        assert [c[6] for c in recorded] == [Path("wta"), None]
        assert "does not match the number of counts files" in capsys.readouterr().out

    def test_existing_outputs_are_skipped_unless_overwriting(
        self, tmp_path, recorded, capsys
    ):
        """Plexes with existing summaries are skipped unless `overwrite` is set."""
        out = self._touch_counts(tmp_path / "o", "counts.1.h5", "counts.1.summary.tsv")
        step5.run(str(out), False, None, False, 0)
        assert recorded == [] and "Skipping existing files" in capsys.readouterr().out
        step5.run(str(out), True, None, False, 0)
        assert len(recorded) == 1 and recorded[0][6] is None
        assert "Overwriting existing files." in capsys.readouterr().out

    def test_non_numeric_plex_names_are_accepted(self, tmp_path, recorded):
        """Non-numeric plex names such as Flex v2 wells are accepted."""
        out = self._touch_counts(tmp_path / "o", "counts.A-A01.h5")
        step5.run(str(out), False, None, False, 0)
        assert recorded[0][9] == "A-A01"

    def test_missing_output_directory_raises(self, tmp_path):
        """A missing output directory raises AssertionError."""
        with pytest.raises(AssertionError):
            step5.run(str(tmp_path / "nope"), False, None, False, 0)


class TestMain:
    def test_arguments_are_forwarded(self, monkeypatch):
        """Options map to `run(output, overwrite, cellranger_output, flatten, reads_per_gapfill)`."""
        calls = []
        monkeypatch.setattr(step5, "run", lambda *a: calls.append(a))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "giftwrap-summarize",
                "-o",
                "out",
                "-f",
                "-wta",
                "a",
                "-wta",
                "b",
                "--reads_per_gapfill",
                "3",
                "--flatten",
            ],
        )
        step5.main()
        monkeypatch.setattr(sys, "argv", ["giftwrap-summarize", "--output", "out"])
        step5.main()
        assert calls == [
            ("out", True, ["a", "b"], True, 3),
            ("out", False, None, False, 0),
        ]

    def test_version(self, monkeypatch, capsys):
        """`--version` prints the package version and exits 0."""
        monkeypatch.setattr(sys, "argv", ["giftwrap-summarize", "--version"])
        with pytest.raises(SystemExit) as exc:
            step5.main()
        assert exc.value.code == 0 and giftwrap.__version__ in capsys.readouterr().out
