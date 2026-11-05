import os
import subprocess
import sys
import textwrap

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import giftwrap.analysis.tools as tl
import support
from support import genotype_frame, to_dense

X = np.array(
    [
        [5, 0, 0, 3, 0],
        [2, 2, 0, 0, 0],
        [0, 0, 0, 0, 4],
        [1, 1, 4, 0, 1],
        [0, 0, 0, 0, 0],
    ]
)
PROBES = ["P1", "P1", "P1", "P2", "P2"]
GAPS = ["A", "C", "G", "T", "TT"]
CELLS = [f"cell{i}" for i in range(5)]


def gapfill(sparse=True, **kwargs):
    return support.make_gapfill_adata(
        X,
        PROBES,
        GAPS,
        sparse=sparse,
        expected={"P1": "C", "P2": "TT"},
        reference={"P1": "A", "P2": "T"},
        **kwargs,
    )


def genotyped(genotypes: dict, index, obs=None):
    adata = ad.AnnData(
        np.zeros((len(index), 1)), obs=pd.DataFrame(obs or {}, index=list(index))
    )
    adata.obsm["genotype"] = pd.DataFrame(genotypes, index=list(index), dtype=object)
    return adata


class TestCollapseGapfills:
    @pytest.mark.parametrize("sparse", [True, False])
    def test_sums_counts_per_probe_and_averages_support(self, sparse):
        """Counts and reads are summed per probe, the gapfill column is dropped and support is averaged."""
        out = tl.collapse_gapfills(gapfill(sparse=sparse))
        assert out.var_names.tolist() == ["P1", "P2"]
        assert (
            "gapfill" not in out.var.columns and "expected_gapfill" in out.var.columns
        )
        np.testing.assert_array_equal(out.X, [[5, 3], [4, 0], [0, 4], [6, 1], [0, 0]])
        np.testing.assert_array_equal(out.layers["total_reads"], 2 * out.X)
        np.testing.assert_allclose(
            out.layers["percent_supporting"],
            [[1 / 3, 1 / 2], [2 / 3, 0], [0, 1 / 2], [1, 1 / 2], [0, 0]],
        )
        assert out.obs_names.tolist() == CELLS

    def test_without_layers(self):
        """Data without layers is collapsed without creating any."""
        adata = gapfill()
        for name in list(adata.layers.keys()):
            del adata.layers[name]
        assert len(tl.collapse_gapfills(adata).layers) == 0


class TestIntersectWta:
    def test_keeps_sorted_shared_cells(self):
        """Both objects are cut to their shared cells in sorted order."""
        wta = ad.AnnData(
            np.ones((3, 2)), obs=pd.DataFrame(index=["cell3", "x", "cell0"])
        )
        w, g = tl.intersect_wta(wta, gapfill())
        assert w.obs_names.tolist() == g.obs_names.tolist() == ["cell0", "cell3"]

    def test_disjoint_cells(self):
        """Objects without shared cells are cut to zero cells."""
        w, g = tl.intersect_wta(
            ad.AnnData(np.ones((1, 1)), obs=pd.DataFrame(index=["x"])), gapfill()
        )
        assert w.n_obs == g.n_obs == 0


class TestCallGenotypes:
    @pytest.mark.parametrize("sparse", [True, False])
    def test_calls_combine_alleles_until_threshold(self, sparse):
        """Top alleles are combined until they reach the threshold; counts, proportions and call arguments are stored."""
        adata = tl.call_genotypes(gapfill(sparse=sparse), threshold=0.5)
        geno = adata.obsm["genotype"]
        assert geno.astype(object).where(geno.notna(), None).values.tolist() == [
            ["A", "T"],
            ["C", None],
            [None, "TT"],
            ["G", "TT"],
            [None, None],
        ]
        assert adata.obsm["genotype_counts"].values.tolist() == [
            [5, 3],
            [2, 0],
            [0, 4],
            [4, 1],
            [0, 0],
        ]
        np.testing.assert_allclose(
            adata.obsm["genotype_proportion"].values,
            [[1, 1], [0.5, 0], [0, 1], [2 / 3, 1], [0, 0]],
        )
        assert all(str(dtype) == "category" for dtype in geno.dtypes)
        assert adata.uns["genotype_call_args"] == {
            "flavor": "basic",
            "threshold": 0.5,
            "cores": 1,
        }

    def test_high_threshold_yields_heterozygous_calls(self):
        """A high threshold combines several alleles into '/'-joined heterozygous calls."""
        geno = tl.call_genotypes(gapfill(), threshold=0.9).obsm["genotype"]
        assert geno.loc["cell1", "P1"] in {"A/C", "C/A"}
        assert geno.loc["cell3", "P1"] in {"G/A/C", "G/C/A"}

    def test_unknown_flavor_is_rejected(self):
        """An unknown `flavor` raises AssertionError."""
        with pytest.raises(AssertionError, match="not recognized"):
            tl.call_genotypes(gapfill(), flavor="bayes")

    @pytest.mark.multiprocess
    def test_parallel_matches_serial_and_non_positive_cores_use_cpu_count(
        self, monkeypatch
    ):
        """Two cores give the same calls as one, and zero cores means all CPUs."""
        serial = tl.call_genotypes(gapfill(), cores=1).obsm["genotype"]
        parallel = tl.call_genotypes(gapfill(), cores=2).obsm["genotype"]
        pd.testing.assert_frame_equal(serial, parallel)
        monkeypatch.setattr(os, "cpu_count", lambda: 2)
        assert (
            tl.call_genotypes(gapfill(), cores=0).uns["genotype_call_args"]["cores"]
            == 2
        )


class TestGenotypeCallJob:
    def test_single_option_probe(self):
        """A probe with one gapfill calls it wherever it was seen and NA elsewhere."""
        calls, n, p = tl._genotype_call_job(["A"], np.array([[3], [0]]), 0.5)
        assert calls.tolist()[0] == "A" and pd.isna(calls[1])
        assert n.tolist() == [3, 0] and p.tolist() == [1.0, 0.0]

    def test_single_detected_and_multi_allele_cells(self):
        """Single-allele cells call that allele; others combine alleles by count until the threshold is met."""
        counts = np.array([[0, 7, 0], [6, 3, 1], [2, 2, 1], [0, 0, 0]])
        calls, n, p = tl._genotype_call_job(["A", "C", "G"], counts, 0.8)
        assert calls[0] == "C" and n[0] == 7 and p[0] == 1.0
        assert calls[1] == "A/C" and n[1] == 9 and p[1] == pytest.approx(0.9)
        assert (
            set(calls[2].split("/")) == {"A", "C"}
            and n[2] == 4
            and p[2] == pytest.approx(0.8)
        )
        assert pd.isna(calls[3]) and n[3] == 0


class TestAnnotateAlleles:
    def test_alleles_are_labelled_against_reference_and_expected(self):
        """Calls are labelled REF, ALT or HET against the panel; missing calls and unknown probes are UNKNOWN."""
        adata = genotyped(
            {"P1": ["A", "C", "A/C", np.nan, "G", ""], "P2": ["T"] * 6},
            CELLS + ["cell5"],
        )
        probe_df = pd.DataFrame(
            {"probe": ["P1"], "reference_gapfill": ["A"], "expected_gapfill": ["C"]}
        )
        out = tl.annotate_alleles(adata, probe_df).obsm["genotype_allele"]
        assert out["P1"].tolist() == ["REF", "ALT", "HET", "UNKNOWN", "ALT", "UNKNOWN"]
        assert out["P2"].tolist() == ["UNKNOWN"] * 6
        assert all(str(d) == "category" for d in out.dtypes)

    def test_strict_unknown_and_no_het_calling(self):
        """Without ALT-for-unrecognized or HET calling, mixed calls get joined labels and unrecognized alleles UNKNOWN."""
        adata = genotyped({"P1": ["A/C", "G", "A/G"]}, CELLS[:3])
        probe_df = pd.DataFrame(
            {"probe": ["P1"], "reference_gapfill": ["A"], "expected_gapfill": ["C"]}
        )
        out = tl.annotate_alleles(
            adata, probe_df, annotate_unrecognized_as_alt=False, call_het=False
        )
        assert out.obsm["genotype_allele"]["P1"].tolist() == [
            "ALT/REF",
            "UNKNOWN",
            "UNKNOWN",
        ]

    def test_probe_annotation_is_inferred_from_uns_then_var(self):
        """Without a probe table, annotations come from `uns['probe_metadata']`, then from `var`."""
        adata = tl.call_genotypes(gapfill())
        adata.uns["probe_metadata"] = pd.DataFrame(
            {
                "probe": ["P1", "P2"],
                "reference_gapfill": ["A", "T"],
                "expected_gapfill": ["C", "TT"],
            }
        )
        assert tl.annotate_alleles(adata).obsm["genotype_allele"]["P2"].tolist() == [
            "REF",
            "UNKNOWN",
            "ALT",
            "ALT",
            "UNKNOWN",
        ]
        del adata.uns["probe_metadata"]
        assert tl.annotate_alleles(adata).obsm["genotype_allele"]["P1"].tolist()[
            :2
        ] == ["REF", "ALT"]

    def test_missing_requirements_raise(self):
        """Missing genotype calls or probe-table columns raise errors."""
        with pytest.raises(AssertionError, match="call_genotypes"):
            tl.annotate_alleles(gapfill())
        with pytest.raises(ValueError, match="missing required columns"):
            tl.annotate_alleles(
                genotyped({"P1": ["A"]}, ["c"]), pd.DataFrame({"probe": ["P1"]})
            )


class TestMutationalBurden:
    def _annotated(self):
        adata = ad.AnnData(np.zeros((4, 1)), obs=pd.DataFrame(index=CELLS[:4]))
        adata.obsm["genotype_allele"] = pd.DataFrame(
            {
                "P1": ["REF", "HET", "ALT", "UNKNOWN"],
                "P2": ["ALT", "REF", "ALT/REF", "UNKNOWN"],
                "P3": ["ALT/ALT", "REF/ALT", np.nan, "UNKNOWN"],
            },
            index=CELLS[:4],
        )
        return adata

    def test_raw_burden_averages_scores_over_valid_sites(self):
        """Raw burden averages allele scores over each cell's sites with a known allele."""
        burden = tl.calculate_mutational_burden(self._annotated(), normalize=False).obs[
            "mutational_burden"
        ]
        assert burden.tolist() == pytest.approx([2 / 3, 1 / 3, 0.75, 0.0])

    def test_normalized_burden_rescales_to_unit_range(self):
        """Normalized burden is min-max scaled to [0, 1]."""
        burden = tl.calculate_mutational_burden(self._annotated()).obs[
            "mutational_burden"
        ]
        assert burden.tolist() == pytest.approx(
            [(2 / 3) / 0.75, (1 / 3) / 0.75, 1.0, 0.0]
        )

    def test_constant_burden_is_left_unchanged_and_annotations_required(self):
        """A constant burden is not rescaled, and missing allele annotations raise AssertionError."""
        adata = ad.AnnData(np.zeros((2, 1)))
        adata.obsm["genotype_allele"] = pd.DataFrame(
            {"P1": ["ALT", "ALT"]}, index=adata.obs_names
        )
        assert tl.calculate_mutational_burden(adata).obs[
            "mutational_burden"
        ].tolist() == [1.0, 1.0]
        with pytest.raises(AssertionError, match="annotate_alleles"):
            tl.calculate_mutational_burden(ad.AnnData(np.zeros((1, 1))))


class TestTransferGenotypes:
    def test_all_wta_cells_present_copies_in_wta_order(self):
        """When every WTA cell was genotyped, genotype tables are copied in WTA order onto the same object."""
        gf = tl.call_genotypes(gapfill())
        wta = ad.AnnData(np.ones((2, 2)), obs=pd.DataFrame(index=["cell3", "cell0"]))
        out = tl.transfer_genotypes(wta, gf)
        assert out is wta
        assert out.obsm["genotype"]["P1"].tolist() == ["G", "A"]
        assert out.obsm["genotype_counts"]["P1"].tolist() == [4, 5]

    def test_missing_cells_are_padded_and_reordered(self):
        """Cells without genotypes get NA rows, and the cells are sorted."""
        gf = tl.call_genotypes(gapfill())
        wta = ad.AnnData(
            np.ones((3, 2)), obs=pd.DataFrame(index=["cell2", "zzz", "cell0"])
        )
        out = tl.transfer_genotypes(wta, gf)
        assert out.obs_names.tolist() == ["cell0", "cell2", "zzz"]
        assert out.obsm["genotype"]["P2"].astype(object).where(
            out.obsm["genotype"]["P2"].notna(), None
        ).tolist() == ["T", "TT", None]
        assert pd.isna(out.obsm["genotype_proportion"].loc["zzz", "P1"])

    def test_requires_genotypes(self):
        """Gapfill data without genotype calls raises AssertionError."""
        with pytest.raises(AssertionError):
            tl.transfer_genotypes(ad.AnnData(np.ones((1, 1))), gapfill())


class TestEncodingAndDistances:
    def test_encoded_matrix_has_one_indicator_column_per_allele(self):
        """Each allele becomes an indicator column; composite calls set several and missing calls are NaN."""
        adata = genotyped(
            {"P1": ["A", "C", "A/C", np.nan], "P2": [np.nan] * 4}, CELLS[:4]
        )
        vectors, labels, alleles = tl._encoded_genotype_matrix(adata)
        assert labels == ["P1", "P2"]
        assert sorted(alleles[0].tolist()) == ["A", "C"] and len(alleles[1]) == 0
        cols = {a: i for i, a in enumerate(alleles[0])}
        assert vectors.shape == (4, 2)
        assert [vectors[r, cols["A"]] for r in range(3)] == [1, 0, 1]
        assert [vectors[r, cols["C"]] for r in range(3)] == [0, 1, 1]
        assert np.isnan(vectors[3]).all()

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="str.split on categorical genotypes yields strings like \"['A']\" instead of allele lists"
    )
    def test_encoding_accepts_categorical_genotypes_from_call_genotypes(self):
        """Categorical genotypes from `call_genotypes` are split into allele lists."""
        adata = tl.call_genotypes(gapfill())
        _, _, alleles = tl._encoded_genotype_matrix(adata)
        assert sorted(alleles[1].tolist()) == ["T", "TT"]

    def test_nan_aware_distance(self):
        """Distance ignores NaN features and is NaN when no features are shared."""
        assert np.isnan(tl._nan_aware_distance(np.array([np.nan]), np.array([1.0])))
        assert tl._nan_aware_distance(
            np.array([1.0, np.nan, 0.0]), np.array([0.0, 1.0, 0.0])
        ) == pytest.approx(0.5)

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="distance is scaled by the number of missing features, so complete profiles get 0"
    )
    def test_fully_observed_different_profiles_have_positive_distance(self):
        """Fully observed but different profiles have a positive distance."""
        assert tl._nan_aware_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) > 0

    def test_vectorized_matrix_matches_pairwise_definition(self):
        """The vectorized distance matrix matches the pairwise definition, with zeros on the diagonal."""
        vectors = np.array(
            [
                [1.0, 0.0, np.nan],
                [0.0, 1.0, 1.0],
                [np.nan, np.nan, np.nan],
                [1.0, np.nan, 0.0],
            ]
        )
        fast = tl._compute_nan_aware_dist_matrix(vectors)
        for i in range(4):
            for j in range(4):
                expected = (
                    0.0 if i == j else tl._nan_aware_distance(vectors[i], vectors[j])
                )
                np.testing.assert_allclose(
                    fast[i, j], expected, rtol=1e-6, equal_nan=True
                )

    def test_custom_metric_uses_sklearn_pairwise_distances(self):
        """A custom distance function gives a square matrix through sklearn's pairwise distances."""
        vectors = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
        out = tl._compute_nan_aware_dist_matrix(
            vectors, distance_func=lambda a, b: float(np.abs(a - b).sum())
        )
        assert out.shape == (2, 2) and out[0, 1] == 0.0

    def test_numba_metric_ignores_masked_features(self):
        """The numba metric ignores masked features and is NaN when nothing can be compared."""
        metric = tl._nan_aware_metric_encoded
        assert metric(
            np.array([1.0, 0.0, 1.0, 1.0]), np.array([0.0, 1.0, 1.0, 0.0])
        ) == pytest.approx(1.0)
        assert np.isnan(
            metric(np.array([1.0, 0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0, 1.0]))
        )


class TestGenotypeConnectivity:
    def _adata(self):
        n = 12
        index = [f"c{i}" for i in range(n)]
        return genotyped(
            {"P1": ["A"] * 6 + ["C"] * 6, "P2": ["T"] * 5 + [np.nan] + ["TT"] * 6},
            index,
        )

    def test_builds_neighbors_graph(self):
        """A kNN graph is built from the encoded genotypes and stored with its parameters."""
        adata = tl.genotype_connectivity(self._adata(), n_neighbors=3)
        assert adata.obsp["genotype_connectivity_connectivities"].shape == (12, 12)
        assert adata.uns["genotype_connectivity"]["params"]["n_neighbors"] == 3
        assert adata.obsm["_genotype_encoded"].shape == (12, 8)

    def test_python_fallback_metric_without_numba(self, monkeypatch):
        """Without numba, the Python metric is used and results go under `key_added`."""
        monkeypatch.setattr(tl, "_nan_aware_metric_encoded", None)
        adata = tl.genotype_connectivity(self._adata(), key_added="geo", n_neighbors=3)
        assert "geo_connectivities" in adata.obsp

    def test_requires_genotypes_and_scanpy(self, monkeypatch):
        """Missing genotype calls raise AssertionError, and a missing scanpy raises ImportError."""
        with pytest.raises(AssertionError):
            tl.genotype_connectivity(ad.AnnData(np.ones((2, 2))))
        monkeypatch.setitem(sys.modules, "scanpy", None)
        with pytest.raises(ImportError, match="scanpy is required"):
            tl.genotype_connectivity(self._adata())


def _clustered():
    index = [f"c{i}" for i in range(12)]
    adata = genotyped(
        {"P1": ["A"] * 6 + ["C"] * 6, "P2": ["T"] * 6 + ["TT"] * 6},
        index,
        obs={"cl": ["a"] * 6 + ["b"] * 6},
    )
    adata.obsm["genotype"].iloc[0, 0] = np.nan
    adata.obsm["genotype"].iloc[7, 1] = np.nan
    return adata


class TestImputeGenotypes:
    @pytest.mark.multiprocess
    def test_missing_calls_are_imputed_from_cluster_neighbours(self):
        """Missing calls are imputed from neighbours in the same cluster, with certainty scores."""
        adata = tl.impute_genotypes(_clustered(), "cl", k=3, hold_out=0.0, cores=2)
        imputed = adata.obsm["genotype_imputed"]
        assert imputed.loc["c0", "P1"] == "A" and imputed.loc["c7", "P2"] == "TT"
        assert imputed["P1"].tolist() == ["A"] * 6 + ["C"] * 6
        assert adata.obsm["genotype_imputed_certainty"].loc["c0", "P1"] == 1.0
        assert "imputation_accuracy" not in adata.uns

    @pytest.mark.multiprocess
    def test_hold_out_reports_accuracy(self):
        """A hold-out percentage stores an imputation accuracy in [0, 1]."""
        np.random.seed(3)
        adata = tl.impute_genotypes(_clustered(), "cl", k=3, hold_out=5.0, cores=2)
        assert 0.0 <= adata.uns["imputation_accuracy"] <= 1.0

    def test_requires_genotypes(self):
        """Data without genotype calls raises ValueError."""
        with pytest.raises(ValueError, match="call_genotypes"):
            tl.impute_genotypes(ad.AnnData(np.ones((1, 1))), "cl")

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=ValueError,
        reason="cores=1 returns a lazy starmap that is consumed twice",
    )
    def test_single_core_imputation(self):
        """Imputation also works on a single core."""
        tl.impute_genotypes(_clustered(), "cl", k=3, hold_out=0.0, cores=1)

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=TypeError,
        reason="imputed alleles cannot be written into categorical genotypes from call_genotypes",
    )
    def test_imputes_genotypes_produced_by_call_genotypes(self):
        """Categorical genotypes from `call_genotypes` can be imputed."""
        adata = _clustered()
        adata.obsm["genotype"] = (
            adata.obsm["genotype"].astype("string").astype("category")
        )
        tl.impute_genotypes(adata, "cl", k=3, hold_out=0.0, cores=2)


class TestImputeWithinCluster:
    def test_cluster_without_any_genotype_returns_empty_frames(self):
        """A cluster without any calls returns all-NA frames and no correct calls."""
        adata = genotyped({"P1": [np.nan, np.nan]}, ["a", "b"], obs={"cl": ["x", "x"]})
        geno, certainty, correct = tl._impute_within_cluster(adata, "cl", "x")
        assert (
            geno.index.tolist() == ["a", "b"]
            and geno.isna().all().all()
            and correct == 0.0
        )

    def test_impute_all_recalls_every_cell(self):
        """`impute_all` recalls every cell, overriding outlier calls."""
        adata = _clustered()
        adata.obsm["genotype"].loc["c1", "P1"] = "C"
        geno, certainty, _ = tl._impute_within_cluster(
            adata, "cl", "a", k=4, impute_all=True
        )
        assert geno.loc["c1", "P1"] == "A"
        assert (certainty.loc[:, "P1"] > 0).all()

    def test_mask_counts_correctly_recovered_calls(self):
        """Masked calls are imputed, and correctly recovered calls are counted."""
        adata = _clustered()
        mask = np.zeros(adata.obsm["genotype"].shape, dtype=bool)
        mask[2, 0] = mask[3, 1] = True
        geno, _, correct = tl._impute_within_cluster(adata, "cl", "a", k=3, mask=mask)
        assert correct == 2
        assert geno.loc["c2", "P1"] == "A"


class TestOffByOneCorrection:
    @pytest.mark.parametrize("sparse", [True, False])
    def test_truncated_gapfill_is_merged_into_full_length_gapfill(self, sparse):
        """A gapfill one base short of a dominant full-length gapfill is merged into it."""
        adata = support.make_gapfill_adata(
            [[10, 2, 0], [0, 0, 5]], ["P"] * 3, ["ACGT", "ACG", "TTTT"], sparse=sparse
        )
        out = tl.correct_off_by_one_gapfills(adata)
        assert out.var_names.tolist() == ["P|ACGT", "P|TTTT"]
        np.testing.assert_array_equal(to_dense(out.X), [[12, 0], [0, 5]])

    def test_corrections_are_skipped_when_evidence_is_weak(self):
        """Gapfills without clear support for a longer variant are left unchanged."""
        adata = support.make_gapfill_adata(
            [[2, 2, 0, 0], [1, 1, 1, 0], [5, 0, 0, 0]],
            ["P"] * 4,
            ["ACGT", "ACG", "TCG", "GG"],
        )
        np.testing.assert_array_equal(
            to_dense(tl.correct_off_by_one_gapfills(adata).X),
            [[2, 2, 0], [1, 1, 1], [5, 0, 0]],
        )

    def test_only_whitelisted_probes_are_corrected(self):
        """With `probes`, only the listed probes are corrected."""
        adata = support.make_gapfill_adata(
            [[10, 2, 10, 2]], ["P", "P", "Q", "Q"], ["ACGT", "ACG", "ACGT", "ACG"]
        )
        out = tl.correct_off_by_one_gapfills(adata, probes=["Q"])
        np.testing.assert_array_equal(to_dense(out.X), [[10, 2, 12]])

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="redistribution relies on pyfamsa profile alignment, which aborts the interpreter"
    )
    def test_multiple_full_length_candidates_redistribute_counts(self):
        """Counts of truncated gapfills are split among several full-length candidates."""
        code = textwrap.dedent("""
            import numpy as np, anndata as ad, pandas as pd, scipy.sparse as sp
            import giftwrap.analysis.tools as tl
            var = pd.DataFrame({"probe": ["P"] * 4, "gapfill": ["ACGTA", "TCGTA", "ACGT", "TCGT"]},
                               index=["P|ACGTA", "P|TCGTA", "P|ACGT", "P|TCGT"])
            out = tl.correct_off_by_one_gapfills(ad.AnnData(sp.csr_matrix(np.array([[6.0, 4, 2, 2]])), var=var))
            assert out.var_names.tolist() == ["P|ACGTA", "P|TCGTA"] and out.X.sum() == 14
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, timeout=120
        )
        assert result.returncode == 0, result.stderr.decode()[-500:]

    @pytest.mark.multiprocess
    def test_parallel_chunks_are_recombined(self):
        """Cells corrected in parallel chunks are recombined in their original order."""
        adata = support.make_gapfill_adata(
            [[10, 2], [3, 1], [0, 4]], ["P", "P"], ["ACGT", "ACG"]
        )
        out = tl.correct_off_by_one_gapfills(adata, cores=2)
        assert out.obs_names.tolist() == adata.obs_names.tolist()
        np.testing.assert_array_equal(to_dense(out.X), [[12, 0], [4, 0], [0, 4]])


class TestGenotypeFrequencies:
    def test_cells_per_genotype_split_composites(self):
        """Cells are counted per allele, with composite calls counting toward each allele."""
        adata = genotyped({"P1": ["A", "A/C", "C", np.nan]}, CELLS[:4])
        name, freqs = tl._generate_genotype_frequencies(adata, "P1", None)
        assert name == "Cells with Genotype" and dict(freqs) == {"A": 2, "C": 2}

    def test_umis_per_genotype_split_evenly_for_composites(self):
        """In 'raw' mode, UMIs are counted per allele and split evenly for composite calls."""
        adata = genotyped({"P1": ["A", "A/C", np.nan]}, CELLS[:3])
        adata.obsm["genotype_counts"] = pd.DataFrame({"P1": [3, 4, 0]}, index=CELLS[:3])
        name, freqs = tl._generate_genotype_frequencies(adata, "P1", "raw")
        assert name == "UMIs with Genotype" and dict(freqs) == {"A": 5.0, "C": 2.0}

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=TypeError,
        reason="'raw' mode compares pd.NA from call_genotypes with != and fails",
    )
    def test_umis_per_genotype_from_called_genotypes(self):
        """'raw' mode works with genotypes from `call_genotypes`."""
        adata = tl.call_genotypes(gapfill())
        assert (
            tl._generate_genotype_frequencies(adata, "P1", "raw")[0]
            == "UMIs with Genotype"
        )

    def test_without_genotypes_uses_raw_gapfill_counts(self, capsys):
        """Without genotype calls, raw gapfill UMI counts are used and a message is printed."""
        adata = support.make_gapfill_adata(
            [[5, 0], [2, 2], [0, 1]], ["P1", "P1"], ["A", "C"]
        )
        name, freqs = tl._generate_genotype_frequencies(adata, "P1", None)
        assert name == "UMIs with Gapfill"
        assert "Using raw probe frequencies" in capsys.readouterr().out

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="raw gapfill totals pair cell i with gapfill i instead of summing each gapfill column"
    )
    def test_raw_gapfill_totals_sum_each_gapfill(self):
        """Raw gapfill totals sum each gapfill over all cells."""
        adata = support.make_gapfill_adata(
            [[5, 0], [2, 2], [0, 1]], ["P1", "P1"], ["A", "C"]
        )
        assert dict(tl._generate_genotype_frequencies(adata, "P1", None)[1]) == {
            "A": 7,
            "C": 3,
        }

    def test_relative_frequencies_mode_label(self):
        """'genotype' mode labels the axis as relative gapfill frequencies."""
        adata = support.make_gapfill_adata([[5, 0], [2, 2]], ["P1", "P1"], ["A", "C"])
        assert (
            tl._generate_genotype_frequencies(adata, "P1", "genotype")[0]
            == "Relative Gapfill Frequencies"
        )


class TestComputeAlignments:
    def test_invalid_motifs_are_dropped_and_sequences_padded(self):
        """Empty, None and 'nan' motifs are dropped, and the rest are padded with '-' to equal length."""
        ref, alt = tl._compute_alignments(
            {"ACGT": 3, "nan": 1, "": 2, None: 4}, {"AC": 1}, align=False
        )
        assert ref == {"ACGT": 3.0} and alt == {"AC--": 1.0}
        ref, alt = tl._compute_alignments({"ACGT": 3, "A": 1}, None, align=False)
        assert ref == {"ACGT": 3.0, "A---": 1.0} and alt is None

    def test_empty_groups_short_circuit(self):
        """Empty groups are returned as-is without aligning."""
        assert tl._compute_alignments({}, {"AC": 1}) == ({}, {"AC": 1.0})
        assert tl._compute_alignments({}, None) == ({}, None)
        assert tl._compute_alignments({"A": 1}, {}) == ({"A": 1.0}, {})

    def test_pyfamsa_alignment_of_single_group(self):
        """With one group, pyFAMSA aligns its motifs to equal length."""
        ref, alt = tl._compute_alignments({"ACGT": 3, "AGT": 1}, None, align=True)
        assert alt is None and sorted(ref.values()) == [1.0, 3.0]
        assert len({len(k) for k in ref}) == 1 and "ACGT" in ref

    def test_too_few_motifs_skip_alignment(self, capsys):
        """Single motifs are not aligned, and a message is printed."""
        assert tl._compute_alignments({"ACGT": 3}, None, align=True) == (
            {"ACGT": 3.0},
            None,
        )
        assert tl._compute_alignments({"ACGT": 3}, {"ACGT": 2}, align=True) == (
            {"ACGT": 3.0},
            {"ACGT": 2.0},
        )
        assert "Not enough motifs to align" in capsys.readouterr().out

    def test_small_groups_are_aligned_jointly(self):
        """Small groups are aligned together so all motifs share one length."""
        ref, alt = tl._compute_alignments({"ACGT": 3, "AGT": 1}, {"ACT": 2}, align=True)
        lengths = {len(k) for k in [*ref, *alt]}
        assert (
            len(lengths) == 1
            and sorted(ref.values()) == [1.0, 3.0]
            and list(alt.values()) == [2.0]
        )

    def test_missing_pyfamsa_falls_back_to_padding(self, monkeypatch, capsys):
        """Without pyFAMSA, motifs are padded instead and a message is printed."""
        monkeypatch.setitem(sys.modules, "pyfamsa", None)
        assert tl._compute_alignments({"ACGT": 3, "A": 1}, None, align=True) == (
            {"ACGT": 3.0, "A---": 1.0},
            None,
        )
        assert "pyFAMSA is not installed" in capsys.readouterr().out

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="profile-profile alignment (both groups >= 2 motifs) aborts the interpreter in pyfamsa"
    )
    def test_large_groups_are_aligned_as_profiles(self):
        """Groups with two or more motifs each are aligned as profiles to one length."""
        code = textwrap.dedent("""
            import giftwrap.analysis.tools as tl
            ref, alt = tl._compute_alignments({"ACGT": 3, "AGT": 1}, {"ACT": 2, "ACGTT": 1}, align=True)
            assert len({len(k) for k in [*ref, *alt]}) == 1
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, timeout=120
        )
        assert result.returncode == 0, result.stderr.decode()[-500:]
