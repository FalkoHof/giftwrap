import anndata as ad
import numpy as np
import pandas as pd
import pytest

import support
from giftwrap.analysis.preprocess import (
    filter_by_min_pcr_duplicates,
    filter_gapfills,
    filter_genotypes,
)
from support import to_dense

X = [[5, 0, 1], [2, 2, 0], [0, 3, 1]]
TOTAL = [[10, 0, 1], [2, 6, 0], [0, 3, 9]]
PERCENT = [[1.0, 0, 0.2], [0.5, 1.0, 0], [0, 0.4, 1.0]]


def gapfill(sparse):
    return support.make_gapfill_adata(
        X,
        ["P1", "P1", "P2"],
        ["A", "C", "T"],
        sparse=sparse,
        total_reads=TOTAL,
        percent_supporting=PERCENT,
    )


@pytest.mark.parametrize("sparse", [True, False])
class TestFilterGapfills:
    def test_no_thresholds_keep_everything(self, sparse):
        """With every threshold disabled, the counts are unchanged."""
        out = filter_gapfills(gapfill(sparse), min_cells=0)
        np.testing.assert_array_equal(to_dense(out.X), X)

    def test_minimum_supporting_reads_masks_counts(self, sparse):
        """Counts are zeroed where total reads fall below `min_supporting_reads`."""
        out = filter_gapfills(gapfill(sparse), min_cells=0, min_supporting_reads=3)
        np.testing.assert_array_equal(
            to_dense(out.X), [[5, 0, 0], [0, 2, 0], [0, 3, 1]]
        )

    def test_minimum_supporting_percent_masks_counts(self, sparse):
        """Counts are zeroed where support falls below `min_supporting_percent`."""
        out = filter_gapfills(gapfill(sparse), min_cells=0, min_supporting_percent=0.5)
        np.testing.assert_array_equal(
            to_dense(out.X), [[5, 0, 0], [2, 2, 0], [0, 0, 1]]
        )

    def test_minimum_umis_masks_counts(self, sparse):
        """Counts below `min_supporting_umis` are zeroed."""
        out = filter_gapfills(gapfill(sparse), min_cells=0, min_supporting_umis=2)
        np.testing.assert_array_equal(
            to_dense(out.X), [[5, 0, 0], [2, 2, 0], [0, 3, 0]]
        )

    def test_minimum_cells_removes_rare_gapfills(self, sparse):
        """Gapfills detected in fewer than `min_cells` cells are dropped (default 10)."""
        assert filter_gapfills(gapfill(sparse), min_cells=2).var_names.tolist() == [
            "P1|A",
            "P1|C",
            "P2|T",
        ]
        assert filter_gapfills(gapfill(sparse), min_cells=3).n_vars == 0
        assert filter_gapfills(gapfill(sparse)).n_vars == 0

    def test_masks_are_applied_before_counting_cells(self, sparse):
        """Cells are counted after the per-entry masks, so masked entries do not count."""
        out = filter_gapfills(gapfill(sparse), min_cells=2, min_supporting_umis=2)
        assert out.var_names.tolist() == ["P1|A", "P1|C"]


def genotyped():
    index = [f"c{i}" for i in range(5)]
    adata = ad.AnnData(np.zeros((5, 1)), obs=pd.DataFrame(index=index))
    adata.obsm["genotype"] = pd.DataFrame(
        {"P1": ["A", "A", "A", "C", "C"], "P2": ["T", np.nan, "T", "T", "G"]},
        index=index,
        dtype=object,
    )
    adata.obsm["genotype_counts"] = pd.DataFrame(
        {"P1": [3, 1, 5, 2, 4], "P2": [2, 0, 2, 2, 9]}, index=index, dtype=float
    )
    adata.obsm["genotype_proportion"] = pd.DataFrame(
        {"P1": [1.0, 0.9, 0.4, 1.0, 0.8], "P2": [1.0, 0, 1.0, 1.0, 1.0]}, index=index
    )
    return adata


def _calls(adata, probe):
    return (
        adata.obsm["genotype"][probe]
        .where(adata.obsm["genotype"][probe].notna(), None)
        .tolist()
    )


class TestFilterGenotypes:
    def test_per_cell_umi_and_proportion_thresholds_mask_calls_and_statistics(self):
        """Calls below the UMI or proportion threshold are masked in all three obsm tables."""
        out = filter_genotypes(
            genotyped(), min_umis_per_cell=2, min_cells=1, min_proportion=0.5
        )
        assert _calls(out, "P1") == ["A", None, None, "C", "C"]
        assert np.isnan(out.obsm["genotype_counts"].loc["c1", "P1"])
        assert np.isnan(out.obsm["genotype_proportion"].loc["c2", "P1"])
        assert out.obsm["genotype_counts"].loc["c0", "P1"] == 3

    def test_genotypes_seen_in_too_few_cells_are_removed(self):
        """A genotype seen in fewer than `min_cells` cells is masked for that probe."""
        out = filter_genotypes(
            genotyped(), min_umis_per_cell=2, min_cells=2, min_proportion=0.5
        )
        assert _calls(out, "P1") == [None, None, None, "C", "C"]
        assert _calls(out, "P2") == ["T", None, "T", "T", None]

    def test_top_n_keeps_most_frequent_genotypes(self):
        """Only the `top_n` most frequent genotypes per probe are kept."""
        out = filter_genotypes(genotyped(), min_umis_per_cell=0, min_cells=0, top_n=1)
        assert _calls(out, "P1") == ["A", "A", "A", None, None]
        assert _calls(out, "P2") == ["T", None, "T", "T", None]

    def test_requires_genotypes(self):
        """Filtering before `call_genotypes` raises an AssertionError."""
        with pytest.raises(AssertionError, match="call_genotypes"):
            filter_genotypes(ad.AnnData(np.zeros((1, 1))))


class TestFilterByMinPcrDuplicates:
    def _adata(self):
        adata = support.make_gapfill_adata(
            [[3, 1]],
            ["P1", "P1"],
            ["A", "C"],
            extra_layers={"X_pcr_threshold_1": [[3, 1]], "X_pcr_threshold_2": [[2, 0]]},
        )
        adata.uns["max_pcr_duplicates"] = 3
        return adata

    def test_counts_are_replaced_by_the_threshold_layer(self):
        """X is replaced in place by the matching `X_pcr_threshold_<n>` layer."""
        adata = self._adata()
        out = filter_by_min_pcr_duplicates(adata, 2)
        assert out is adata
        np.testing.assert_array_equal(to_dense(out.X), [[2, 0]])

    def test_threshold_above_maximum_is_rejected(self):
        """A threshold above `max_pcr_duplicates` raises ValueError."""
        with pytest.raises(ValueError, match="cannot be greater"):
            filter_by_min_pcr_duplicates(self._adata(), 4)
