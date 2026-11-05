import json
import re

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import giftwrap.analysis.tools as tools
import giftwrap.html_report as hr
import support
from giftwrap.pipeline_utils import write_manifest

PROBES = ["GENEA c.1A>T", "GENEA c.1A>T", "GENEB c.2C>G", "GENEC 0bp"]
GAPS = ["TTG", "TAG", "GCAT", ""]
X = np.array([[2, 1, 0, 1], [0, 0, 3, 1], [1, 0, 0, 0], [0, 0, 0, 0]])
CELLS = [f"{c}-1" for c in support.CELLS[:4]]


def gapfill_adata(x=X, probes=PROBES, gaps=GAPS, **kwargs):
    return support.make_gapfill_adata(
        x, probes, gaps, obs_names=CELLS[: len(x)], **kwargs
    )


class TestSmallHelpers:
    def test_read_asset_returns_vendored_text_or_none(self, tmp_path, monkeypatch):
        """Vendored assets are returned as text; a missing file or asset directory gives None."""
        assert "plotly" in hr._read_asset("plotly.min.js").lower()
        assert hr._read_asset("does-not-exist.js") is None
        monkeypatch.setattr(hr, "_ASSETS_DIR", tmp_path)
        assert hr._read_asset("plotly.min.js") is None

    def test_to_list_flattens_arrays_and_matrices(self):
        """Column vectors and sparse row sums become flat Python lists."""
        assert hr._to_list(np.array([[1], [2]])) == [1, 2]
        assert hr._to_list(sp.csr_matrix([[1, 2], [3, 4]]).sum(axis=1)) == [3, 7]

    @pytest.mark.parametrize(
        ("value", "group_max", "expected"),
        [
            (5, 10, "background: rgba(5,150,105,0.400);"),
            (20, 10, "background: rgba(5,150,105,0.700);"),
            (-1, 10, "background: rgba(5,150,105,0.100);"),
            (5, 0, "background: #F8FAFC;"),
            (5, -1, "background: #F8FAFC;"),
        ],
    )
    def test_magnitude_background_scales_alpha_within_group(
        self, value, group_max, expected
    ):
        """Alpha scales with value / group max within [0.1, 0.7]; a non-positive max gives a neutral background."""
        assert hr._magnitude_bg(value, group_max, hr._GREEN_RGB) == expected

    def test_ratio_background_blends_red_to_green_and_clamps(self):
        """Ratios blend from red at 0 to green at 1 and are clamped to that range."""
        assert hr._ratio_bg(0.0) == "background: rgb(243,179,179);"
        assert hr._ratio_bg(-3) == hr._ratio_bg(0.0)
        assert hr._ratio_bg(5) == hr._ratio_bg(1.0) != hr._ratio_bg(0.5)

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [("5", 0, 5), (7.9, 0, 7), (None, -1, -1), ("x", 3, 3)],
    )
    def test_as_int(self, value, default, expected):
        """Numbers and numeric strings are truncated to int; None or invalid values give the default."""
        assert hr._as_int(value, default) == expected

    @pytest.mark.parametrize(
        ("tech", "label"),
        [
            (None, None),
            ("", None),
            ("Flex", "10x Flex v1"),
            ("Flex-v2-R1", "10x Flex v2 (R1 demux)"),
            ("VisiumHD", "Visium HD"),
            ("Visium-v4", "Visium v4"),
            ("Other", "Other"),
        ],
    )
    def test_technology_label(self, tech, label):
        """Known technologies get display labels; empty values give None and unknown names pass through."""
        assert hr._technology_label(tech) == label


class TestSaturationCurve:
    def test_empty_counts_yield_no_curve(self):
        """All-zero or empty count matrices give an empty curve."""
        assert hr._saturation_curve(np.zeros((3, 2))) == ([], [])
        assert hr._saturation_curve(np.zeros((0, 2))) == ([], [])

    def test_curve_is_seeded_and_bounded(self):
        """The curve is deterministic for sparse and dense input, ends at mean UMIs per cell and stays in [0, 1]."""
        layer = sp.csr_matrix(np.array([[5, 0, 2], [1, 3, 0]]))
        xs, ys = hr._saturation_curve(layer, n_points=10)
        assert (xs, ys) == hr._saturation_curve(layer.toarray(), n_points=10)
        assert len(xs) == len(ys) == 10
        assert xs == sorted(xs) and xs[-1] == pytest.approx(11 / 2)
        assert all(0.0 <= y <= 1.0 for y in ys)


class TestFigures:
    def test_sankey_keeps_only_positive_flows_with_percentages_and_colors(self):
        """Only non-zero flows become links, with percent-of-source values and red tints for filtered reads."""
        fig = hr._fig_sankey({"EXACT": 5, "FILTERED_NO_LHS": 2}, {})
        link = fig["data"][0]["link"]
        labels = fig["data"][0]["node"]["label"]
        pairs = [(labels[s], labels[t]) for s, t in zip(link["source"], link["target"])]
        assert pairs == [
            ("All Reads", "Exact Matches"),
            ("Exact Matches", "Mapped Reads"),
            ("All Reads", "No LHS"),
            ("No LHS", "Filtered"),
        ]
        assert link["value"] == [5, 5, 2, 2]
        assert link["customdata"] == pytest.approx([500 / 7, 100.0, 200 / 7, 100.0])
        assert [c.startswith("rgba(220") for c in link["color"]] == [
            False,
            False,
            True,
            True,
        ]

    def test_barcode_rank_is_sorted_and_subsampled_for_large_inputs(self):
        """Barcodes are ranked by descending UMIs, and large inputs are subsampled but keep the top barcode."""
        small = hr._fig_barcode_rank(gapfill_adata())["data"][0]
        assert small["x"] == [1, 2, 3, 4] and small["y"] == [4, 4, 1, 0]
        rng = np.random.default_rng(1)
        big = ad.AnnData(sp.csr_matrix(rng.integers(0, 5, size=(2500, 3))))
        trace = hr._fig_barcode_rank(big)["data"][0]
        assert len(trace["x"]) < 2500 and trace["x"][0] == 1
        assert trace["y"][0] == big.X.sum(axis=1).max()

    def test_saturation_figure_falls_back_to_raw_layer(self, monkeypatch):
        """If collapsing gapfills fails, the saturation curve uses the uncollapsed counts."""
        assert len(hr._fig_saturation_curve(gapfill_adata())["data"][0]["x"]) == 100

        def boom(adata):
            raise RuntimeError("collapse failed")

        monkeypatch.setattr(tools, "collapse_gapfills", boom)
        assert len(hr._fig_saturation_curve(gapfill_adata())["data"][0]["y"]) == 100

    def test_pcr_histogram(self, tmp_path):
        """Missing or empty reads files give no figure; otherwise duplicates are binned, with a cutoff line when set."""
        assert hr._fig_pcr_histogram(tmp_path / "missing.tsv.gz", 0) == {}
        header = ["a", "b", "c", "d", "e", "dup", "g"]
        assert (
            hr._fig_pcr_histogram(
                support.write_tsv(tmp_path / "empty.tsv", header, []), 0
            )
            == {}
        )
        reads = support.write_tsv(
            tmp_path / "r.tsv",
            header,
            [[0, 0, 1, "U", "A", 1, 1.0], [0, 0, 1, "U", "A", 9, 1.0], ["x"]],
        )
        fig = hr._fig_pcr_histogram(reads, 0)
        assert sum(fig["data"][0]["y"]) == 2 and fig["layout"]["shapes"] == []
        assert hr._fig_pcr_histogram(reads, 3)["layout"]["shapes"][0][
            "x0"
        ] == pytest.approx(np.log10(4))

    def test_umis_and_cells_histograms_exclude_controls_where_relevant(self):
        """The UMIs-per-cell histogram covers every cell, while cells-per-gapfill excludes 0bp controls."""
        assert sum(hr._fig_umis_per_cell(gapfill_adata())["data"][0]["y"]) == 4
        assert sum(hr._fig_cells_per_gapfill(gapfill_adata())["data"][0]["y"]) == 3

    def test_probe_prevalence_bars(self):
        """The top-N probe/gapfill pairs are shown by ascending percent of cells, with hover details."""
        fig = hr._fig_probe_boxplot(gapfill_adata(), top_n=2)
        trace = fig["data"][0]
        assert trace["y"] == ["GENEB c.2C>G · GCAT", "GENEA c.1A>T · TTG"]
        assert trace["x"] == pytest.approx([25.0, 50.0])
        assert trace["customdata"][1] == ["GENEA c.1A>T", "TTG", 2]

    def test_probe_prevalence_edge_cases(self):
        """Long gapfills are truncated, missing gapfills fall back to feature names, and empty inputs give no figure."""
        long_gap = "A" * 40
        adata = gapfill_adata(np.ones((1, 1)), ["P1"], [long_gap])
        assert hr._fig_probe_boxplot(adata)["data"][0]["y"] == [
            "P1 · " + "A" * 27 + "…"
        ]
        no_probe = ad.AnnData(np.ones((2, 2)))
        assert hr._fig_probe_boxplot(no_probe) == {}
        no_gapfill = gapfill_adata()
        del no_gapfill.var["gapfill"]
        assert (
            hr._fig_probe_boxplot(no_gapfill)["data"][0]["customdata"][-1][1]
            == "GENEA c.1A>T|TTG"
        )
        assert hr._fig_probe_boxplot(gapfill_adata()[:0]) == {}

    def test_reads_per_gapfill_ranks_variants_with_median_line(self):
        """Variants are ranked by total reads with a median line; without `total_reads` there is no figure."""
        fig = hr._fig_reads_per_gapfill(gapfill_adata())
        bars, median = fig["data"]
        assert bars["y"] == [6, 6, 2] and sorted(bars["customdata"]) == [
            "GENEA c.1A>T",
            "GENEA c.1A>T",
            "GENEB c.2C>G",
        ]
        assert bars["customdata"][-1] == "GENEA c.1A>T"
        assert median["y"] == [6.0, 6.0]
        dense = gapfill_adata(sparse=False)
        assert hr._fig_reads_per_gapfill(dense)["data"][0]["y"] == [6, 6, 2]
        without = gapfill_adata()
        del without.layers["total_reads"]
        assert hr._fig_reads_per_gapfill(without) == {}

    def test_reads_per_gapfill_subsamples_large_panels(self):
        """Large panels are subsampled while keeping the top-ranked variant."""
        rng = np.random.default_rng(0)
        n = 2500
        adata = support.make_gapfill_adata(
            rng.integers(1, 9, size=(3, n)), [f"P{i}" for i in range(n)], ["A"] * n
        )
        bars = hr._fig_reads_per_gapfill(adata)["data"][0]
        assert len(bars["x"]) < n and bars["x"][0] == 0


class TestFitLine:
    def test_too_few_points(self):
        """Fewer than two finite points give an empty fit with R² of 0."""
        assert hr._fit_line([1.0], [2.0]) == ([], [], 0.0)
        assert hr._fit_line([np.nan, 1.0], [1.0, np.inf]) == ([], [], 0.0)

    def test_constant_x(self):
        """Constant x gives a flat line at mean y, with R² of 1 only when y is constant too."""
        assert hr._fit_line([2, 2, 2], [1, 1, 1]) == ([2.0, 2.0], [1.0, 1.0], 1.0)
        assert hr._fit_line([2, 2], [1, 3]) == ([2.0, 2.0], [2.0, 2.0], 0.0)

    def test_perfect_and_noisy_lines(self):
        """A perfect line is evaluated at unique x with R² of 1; noisy data gives R² in [0, 1)."""
        xs, ys, r2 = hr._fit_line([0, 1, 2, 2], [1, 3, 5, 5])
        assert (
            xs == [0, 1, 2]
            and ys == pytest.approx([1, 3, 5])
            and r2 == pytest.approx(1.0)
        )
        assert 0.0 <= hr._fit_line([0, 1, 2, 3], [0, 3, 1, 2])[2] < 1.0


class TestWtaCorrelation:
    def _wta(self, barcodes, genes, matrix):
        return ad.AnnData(
            sp.csr_matrix(np.asarray(matrix, dtype=float)),
            obs=pd.DataFrame(index=barcodes),
            var=pd.DataFrame(index=genes),
        )

    def test_without_wta_or_shared_genes(self):
        """Without WTA data or shared genes, no figures or statistics are produced."""
        assert hr._fig_wta_correlation(gapfill_adata(), None) == {
            "figs": [],
            "stats": None,
        }
        wta = self._wta(CELLS, ["X1"], [[1], [2], [3], [4]])
        assert hr._fig_wta_correlation(gapfill_adata(), wta) == {
            "figs": [],
            "stats": None,
        }

    def test_correlations_use_inferred_genes_and_normalized_barcodes(self):
        """Genes are inferred from probe names and plex suffixes are stripped from barcodes before correlating."""
        gf = gapfill_adata()
        gf.obs_names = [f"{c}{support.FLEX_BC1}-1" for c in support.CELLS[:4]]
        wta = self._wta(CELLS, ["GENEA", "GENEB"], [[3, 0], [0, 4], [1, 2], [0, 5]])
        result = hr._fig_wta_correlation(gf, wta)
        assert gf.var["gene"].tolist() == ["GENEA", "GENEA", "GENEB", "GENEC"]
        assert len(result["figs"]) == 2
        sc, pb = result["stats"]["single_cell"], result["stats"]["pseudobulk"]
        assert sc["zero_frac"] == pytest.approx(2 / 5)
        assert pb["zero_frac"] == 0.0
        assert -1.0 <= sc["rho"] <= 1.0
        assert result["figs"][0]["data"][0]["customdata"][0] == [
            "GENEA c.1A>T",
            "GENEA",
        ]


class TestHeroMetrics:
    def _by_label(self, metrics):
        return {m["label"]: m for m in metrics}

    def test_metrics_from_run_statistics(self):
        """Eleven headline metrics are formatted from the FASTQ and count statistics."""
        fastq = {
            "TOTAL_READS": 200,
            "PROBE_CONTAINING_READS": 50,
            "PROBES_ENCOUNTERED": 3,
            "POSSIBLE_PROBES": 4,
            "R1_CYCLES": 28,
            "R2_CYCLES": "0",
        }
        counts = {
            "SEQUENCING_SATURATION": 0.5,
            "GAPFILL_CONTAINING_CELLS": 3,
            "TOTAL_CELLS": 4,
            "UMIS_PER_CELL_MEAN": 2.25,
            "TOTAL_UMIS": 9,
        }
        m = self._by_label(hr._build_hero_metrics(fastq, counts, gapfill_adata()))
        assert len(m) == 11
        assert m["Reads Mapped"]["value"] == "25.0%"
        assert m["Sequencing Saturation"]["value"] == "50.0%"
        assert m["Cells with Probes / Total Cells"]["value"] == "75.0%"
        assert m["Encountered / Possible Probes"]["value"] == "3 / 4"
        assert m["Encountered / Possible Probes"]["sub"].endswith("(75.0%)")
        assert m["Probes / Cell"]["value"] == "1.5"
        assert m["Mean UMIs / Cell"]["value"] == "2.2"
        assert m["R1 Cycles"]["value"] == "28" and m["R2 Cycles"]["value"] == "n/a"
        assert m["Total UMIs"]["value"] == "9" and m["Total Reads"]["value"] == "200"

    def test_metrics_without_inputs_degrade_to_neutral_values(self):
        """Missing statistics give 0 or 'n/a' values instead of errors."""
        m = self._by_label(hr._build_hero_metrics({}, {}))
        assert m["Reads Mapped"]["value"] == "0.0%"
        assert m["Cells with Probes / Total Cells"]["value"] == "n/a"
        assert m["Probes / Cell"]["value"] == "0.0"
        assert m["R1 Cycles"]["value"] == "n/a"
        assert (
            m["Encountered / Possible Probes"]["sub"] == "probes detected / panel size"
        )


class TestCommandBlock:
    def test_no_invocation_renders_nothing(self):
        """No invocations, or an empty command, render nothing."""
        assert hr._command_block_html([]) == ""
        assert hr._command_block_html([{"command": ""}]) == ""

    def test_latest_command_is_escaped_with_metadata(self):
        """The latest command is HTML-escaped and shown with its version and timestamp."""
        out = hr._command_block_html(
            [
                {
                    "command": "giftwrap -o '<x>'",
                    "giftwrap_version": "1.0",
                    "timestamp": "2026-01-02T03:04:05+00:00",
                }
            ]
        )
        assert "<code>giftwrap -o &#x27;&lt;x&gt;&#x27;</code>" in out
        assert "v1.0 &middot; 2026-01-02 03:04" in out
        assert "help-btn" not in out

    def test_earlier_runs_are_listed_in_tooltip(self):
        """Earlier runs are listed in a help tooltip under the latest command."""
        out = hr._command_block_html(
            [
                {"command": "giftwrap a", "timestamp": "2026-01-01T00:00:00"},
                {"command": "giftwrap b"},
            ]
        )
        assert "<code>giftwrap b</code>" in out
        assert (
            "produced by 2 giftwrap runs" in out
            and "2026-01-01 00:00  giftwrap a" in out
        )


def _cycle(fastq, config):
    flags = []
    hr._cycle_flags(fastq, config, lambda *a: flags.append(a))
    return flags


class TestCycleFlags:
    def test_no_configuration_means_no_checks(self):
        """Without a run configuration, no cycle checks are made."""
        assert (
            _cycle({"R2_CYCLES": 10}, None) == []
            and _cycle({"R2_CYCLES": 10}, {}) == []
        )

    @pytest.mark.parametrize("tech", ["Flex", "Flex-v2"])
    def test_multiplexed_r2_too_short_for_sample_barcode(self, tech):
        """Multiplexed Flex runs error when R2 cannot reach the sample barcode; Flex v2 suggests Flex-v2-R1."""
        config = {
            "technology": tech,
            "multiplexed": True,
            "barcodes": ["1", "2"],
            "min_r2_for_probe_barcode": 80,
        }
        [(level, label, title, message)] = _cycle({"R2_CYCLES": 50}, config)
        assert (level, label, title) == (
            "error",
            "R2 Cycles",
            "R2 too short to read the sample barcode",
        )
        assert "2 barcodes" in message and "at least 80 R2 cycles" in message
        assert ("Flex-v2-R1 instead" in message) == (tech == "Flex-v2")

    def test_r1_too_short_when_demultiplexing_from_r1(self):
        """Flex-v2-R1 runs error when R1 cannot reach the probe barcode."""
        config = {
            "technology": "Flex-v2-R1",
            "multiplexed": True,
            "multiplex": 1,
            "min_r1_for_probe_barcode": 50,
        }
        flags = _cycle({"R1_CYCLES": 28, "R2_CYCLES": 90}, config)
        assert [(f[0], f[1]) for f in flags] == [("error", "R1 Cycles")]
        assert "(1 barcode)" in flags[0][3]

    @pytest.mark.parametrize(
        ("extra", "flagged"),
        [
            ({}, True),
            ({"skip_constant_seq": True}, False),
            ({"trim_probes": 40}, False),
            ({"expected_gap_length": 0}, False),
            ({"expected_gap_length": -1}, True),
        ],
    )
    def test_short_r2_without_truncation_handling_warns(self, extra, flagged):
        """Short R2 warns unless constant skipping, probe trimming or a non-negative expected gap applies."""
        flags = _cycle({"R2_CYCLES": 50}, {"technology": "Flex"} | extra)
        assert (
            [f[:3] for f in flags]
            == [("warning", "R2 Cycles", "Short R2 without truncated-read handling")]
        ) is flagged

    def test_long_reads_and_non_flex_technologies_pass(self):
        """Long enough R2 reads and non-Flex technologies raise no cycle flags."""
        assert _cycle({"R2_CYCLES": 90}, {"technology": "Flex"}) == []
        assert _cycle({"R2_CYCLES": 40}, {"technology": "VisiumHD"}) == []


def _flags(fastq=None, counts=None, adata=None, wta=None, config=None):
    base_counts = {"SEQUENCING_SATURATION": 0.9}
    return hr._evaluate_flags(
        fastq or {}, base_counts | (counts or {}), adata, wta, config
    )


def _titles(flags):
    return [(f["level"], f["title"]) for f in flags]


class TestEvaluateFlags:
    @pytest.mark.parametrize(
        ("encountered", "expected"),
        [
            (0, [("error", "No probes detected")]),
            (4, [("warning", "Low probe detection rate")]),
            (6, []),
        ],
    )
    def test_probe_detection(self, encountered, expected):
        """No detected probes is an error, and a low probe detection rate is a warning."""
        assert (
            _titles(_flags({"POSSIBLE_PROBES": 10, "PROBES_ENCOUNTERED": encountered}))
            == expected
        )

    @pytest.mark.parametrize(
        ("mapped", "expected"),
        [
            (20, [("error", "Very low read mapping rate")]),
            (40, [("warning", "Low read mapping rate")]),
            (60, []),
        ],
    )
    def test_read_mapping(self, mapped, expected):
        """Low read mapping rates warn or error and link to the overview Sankey plot."""
        flags = _flags({"TOTAL_READS": 100, "PROBE_CONTAINING_READS": mapped})
        assert _titles(flags) == expected
        assert all(
            f["tab"] == "overview" and f["plot_id"] == "plot-sankey-overview"
            for f in flags
        )

    @pytest.mark.parametrize(
        ("saturation", "expected"),
        [
            (0.2, [("error", "Very low sequencing saturation")]),
            (0.4, [("warning", "Low sequencing saturation")]),
            (0.6, []),
        ],
    )
    def test_saturation(self, saturation, expected):
        """Low sequencing saturation warns or errors depending on how low it is."""
        assert _titles(_flags(counts={"SEQUENCING_SATURATION": saturation})) == expected

    @pytest.mark.parametrize(
        ("with_probes", "expected"),
        [
            (5, [("error", "Very few cells have any gapfill signal")]),
            (20, [("warning", "Few cells have gapfill signal")]),
            (50, []),
            (100, []),
        ],
    )
    def test_cells_with_probes(self, with_probes, expected):
        """A low share of cells with gapfill signal warns or errors."""
        assert (
            _titles(
                _flags(
                    counts={"TOTAL_CELLS": 100, "GAPFILL_CONTAINING_CELLS": with_probes}
                )
            )
            == expected
        )

    def test_detection_breadth_and_undetected_controls(self):
        """Variants seen in few cells flag poor or limited breadth, and undetected 0bp controls are named."""
        singleton = gapfill_adata(
            np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        )
        flags = _flags(adata=singleton)
        assert _titles(flags) == [
            ("error", "Poor detection breadth of gapfill variants"),
            ("error", "Undetected 0bp control probe(s)"),
        ]
        assert "GENEC 0bp" in flags[1]["message"]
        mixed = gapfill_adata(
            np.array([[1, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
        )
        assert _titles(_flags(adata=mixed)) == [
            ("warning", "Limited detection breadth of gapfill variants")
        ]
        broad = gapfill_adata(np.array([[1, 1, 1, 1]] * 4))
        assert _flags(adata=broad) == []
        assert _flags(adata=gapfill_adata()[:, :0]) == []

    def test_breadth_is_skipped_without_detected_real_variants(self):
        """Breadth is not judged without detected real variants, but undetected controls are still flagged."""
        controls_only = gapfill_adata(np.ones((2, 1)), ["GENEC 0bp"], [""])
        assert _flags(adata=controls_only) == []
        undetected = gapfill_adata(
            np.zeros((2, 2)), ["GENEA x", "GENEC 0bp"], ["A", ""]
        )
        assert _titles(_flags(adata=undetected)) == [
            ("error", "Undetected 0bp control probe(s)")
        ]

    @pytest.mark.parametrize(
        ("stats", "expected"),
        [
            ({"rho": -0.1, "zero_frac": 0.0}, "error"),
            ({"rho": 0.5, "zero_frac": 0.8}, "error"),
            ({"rho": 0.01, "zero_frac": 0.0}, "warning"),
            ({"rho": 0.5, "zero_frac": 0.6}, "warning"),
            ({"rho": 0.5, "zero_frac": 0.1}, None),
        ],
    )
    def test_wta_correlation(self, stats, expected):
        """Weak or negative single-cell WTA correlation, or many zeros, is flagged as a warning or error."""
        flags = _flags(wta={"single_cell": stats, "pseudobulk": None})
        assert [f["level"] for f in flags] == ([expected] if expected else [])
        if expected:
            assert (
                flags[0]["title"] == "Weak single-cell correlation"
                and flags[0]["plot_id"] == "plot-wta-sc"
            )

    def test_cycle_flags_are_included(self):
        """Read-cycle flags are appended and linked to their headline metric."""
        flags = _flags({"R2_CYCLES": 40}, config={"technology": "Flex"})
        assert flags[-1]["hero_label"] == "R2 Cycles"


class TestFlagCards:
    def test_all_clear_banner(self):
        """Without flags, an all-clear banner is shown."""
        assert "All QC checks passed" in hr._flag_cards_html([])

    def test_errors_first_with_jump_buttons_only_when_linked(self):
        """Errors come before warnings, and only flags linked to a plot get a jump button."""
        flags = [
            {
                "level": "warning",
                "title": "W",
                "message": "w",
                "tab": None,
                "plot_id": None,
            },
            {
                "level": "error",
                "title": "E",
                "message": "e",
                "tab": "overview",
                "plot_id": "plot-x",
            },
        ]
        html = hr._flag_cards_html(flags)
        assert html.index("flag-card error") < html.index("flag-card warning")
        assert (
            html.count("flag-jump") == 1
            and "jumpToPlot('overview','plot-x','error')" in html
        )

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="flag titles/messages (which embed probe names) are inserted into HTML unescaped"
    )
    def test_messages_are_html_escaped(self):
        """Flag titles and messages are HTML-escaped before insertion."""
        html = hr._flag_cards_html(
            [{"level": "error", "title": "T", "message": "<script>alert(1)</script>"}]
        )
        assert "<script>" not in html


class TestRowStyles:
    def test_fastq_rows_are_tinted_by_group(self):
        """FASTQ rows are tinted green or red relative to their group maximum; totals use the info background."""
        fastq = {
            "EXACT": 10,
            "CORRECTED_LHS": 5,
            "FILTERED_NO_LHS": 4,
            "FILTERED_NO_RHS": 2,
        }
        assert hr._fastq_row_style("CORRECTED_LHS", 5, fastq) == hr._magnitude_bg(
            5, 10, hr._GREEN_RGB
        )
        assert hr._fastq_row_style("FILTERED_NO_RHS", 2, fastq) == hr._magnitude_bg(
            2, 4, hr._RED_RGB
        )
        assert hr._fastq_row_style("TOTAL_READS", 100, fastq) == hr._INFO_BG

    def test_counts_rows_use_ratio_coloring(self):
        """Ratio rows use red-to-green coloring, and other count rows use the info background."""
        counts = {"TOTAL_CELLS": 4}
        assert hr._counts_row_style(
            "GAPFILL_CONTAINING_CELLS", 3, counts
        ) == hr._ratio_bg(0.75)
        assert hr._counts_row_style("GAPFILL_CONTAINING_CELLS", 3, {}) == hr._ratio_bg(
            0.0
        )
        assert hr._counts_row_style(
            "SEQUENCING_SATURATION", 0.4, counts
        ) == hr._ratio_bg(0.4)
        assert hr._counts_row_style("TOTAL_UMIS", 9, counts) == hr._INFO_BG


@pytest.fixture
def report_dir(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    pd.DataFrame(
        {
            "metric": ["TOTAL_READS", "PROBE_CONTAINING_READS", "EXACT", "R2_CYCLES"],
            "value": [100, 60, 60, 90],
        }
    ).to_csv(out / "fastq_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "statistic": ["TOTAL_CELLS", "SEQUENCING_SATURATION", "UMIS_PER_CELL_MEAN"],
            "value": [4, 0.6, 2.25],
        }
    ).to_csv(out / "counts.1.summary.tsv", sep="\t", index=False)
    support.write_tsv(
        out / "probe_reads.tsv.gz", list("abcdefg"), [[0, 0, 1, "U", "TTG", 3, 1.0]]
    )
    return out


def _report(out, name="counts.1.summary.html", adata=None, **kwargs):
    target = out / name
    hr.make_html_report(
        target,
        gapfill_adata(),
        adata,
        out / "fastq_metrics.tsv",
        out / "counts.1.summary.tsv",
        out / "probe_reads.tsv.gz",
        **kwargs,
    )
    return target.read_text()


def _figs(html):
    return json.loads(re.search(r"const FIGS = (\{.*?\});\n", html, re.S).group(1))


class TestMakeHtmlReport:
    def test_self_contained_report_with_metrics_sidecar(self, report_dir):
        """The report inlines Plotly and fonts, includes the core figures and writes a metrics CSV."""
        html = _report(report_dir, filter_cutoff=2)
        assert hr._read_asset("plotly.min.js")[:200] in html
        assert (
            '<script src="https://cdn.plot.ly' not in html
            and '<link href="https://fonts.googleapis.com' not in html
        )
        assert "<h1>Plex 1</h1>" in html
        assert "<b>Chemistry</b> 10x Flex</span>" in html
        assert 'data-tab="wta"' not in html
        assert set(_figs(html)) == {
            "sankey",
            "barcode_rank",
            "saturation",
            "pcr",
            "umis_per_cell",
            "cells_per_gapfill",
            "probe_boxplot",
            "reads_per_gapfill",
        }
        metrics = pd.read_csv(report_dir / "counts.1.metrics.csv")
        assert metrics.columns.tolist() == ["category", "metric", "value"]
        assert metrics.category.tolist() == ["counts"] * 3 + ["fastq"] * 4
        assert 'download="counts.1.metrics.csv"' in html

    def test_run_configuration_invocations_and_identifiers(self, report_dir):
        """Sample, iLab and plex ids, the configured chemistry and the invocation appear in the report."""
        write_manifest(report_dir, support.manifest_df(), {"technology": "Visium-v2"})
        (report_dir / "pipeline_invocation.json").write_text(
            json.dumps([{"command": "giftwrap -o run"}])
        )
        html = _report(
            report_dir, name="report.html", sample_id="LIB-7", ilab="IL-9", plex_id="2"
        )
        assert "<h1>LIB-7 — Plex 2</h1>" in html
        assert "<b>Library</b> LIB-7" in html and "<b>iLab</b> IL-9" in html
        assert "<b>Chemistry</b> Visium v2</span>" in html
        assert "<code>giftwrap -o run</code>" in html
        assert (report_dir / "2.metrics.csv").exists()

    def test_plex_is_inferred_from_simple_file_names(self, report_dir):
        """Without a plex in the file name, the title reads 'Plex report'."""
        assert "<h1>Plex report</h1>" in _report(report_dir, name="report.html")

    def test_wta_tab_and_spatial_chemistry_guess(self, report_dir):
        """WTA data adds a correlation tab, and spatial WTA without a config is labelled Visium HD."""
        wta = ad.AnnData(
            sp.csr_matrix(np.ones((4, 2))),
            obs=pd.DataFrame(index=CELLS),
            var=pd.DataFrame(index=["GENEA", "GENEB"]),
        )
        wta.obsm["spatial"] = np.zeros((4, 2))
        html = _report(report_dir, adata=wta)
        assert 'data-tab="wta"' in html and {"wta_sc", "wta_pb"} <= set(_figs(html))
        assert "<b>Chemistry</b> Visium HD</span>" in html

    def test_missing_inputs_and_assets_fall_back(
        self, report_dir, tmp_path, monkeypatch, capsys
    ):
        """Unreadable inputs only print messages, and missing assets fall back to CDN links."""

        def unreadable(path):
            raise PermissionError("denied")

        monkeypatch.setattr(hr, "read_pipeline_invocations", unreadable)
        monkeypatch.setattr(hr, "_ASSETS_DIR", tmp_path / "no-assets")
        target = report_dir / "counts.1.summary.html"
        hr.make_html_report(
            target,
            gapfill_adata(),
            None,
            report_dir / "nope.tsv",
            report_dir / "nope2.tsv",
            report_dir / "nope3.tsv",
        )
        html = target.read_text()
        out = capsys.readouterr().out
        assert (
            "could not read fastq_metrics.tsv" in out
            and "could not read summary.tsv" in out
        )
        assert "could not read run configuration" in out
        assert "could not read pipeline_invocation.json: denied" in out
        assert '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js">' in html
        assert '<link href="https://fonts.googleapis.com' in html
