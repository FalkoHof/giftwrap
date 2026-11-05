import anndata as ad
import numpy as np
import pandas as pd
import pytest

import giftwrap.analysis.plots as pl
import support


def gapfill(**kwargs):
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 5, size=(8, 4))
    return support.make_gapfill_adata(
        counts,
        ["P1", "P1", "P1", "P2"],
        ["AC", "AG", "ACT", "T"],
        obs={"group": ["g1"] * 4 + ["g2"] * 4},
        **kwargs,
    )


def genotyped(genotypes):
    adata = gapfill()
    adata.obsm["genotype"] = pd.DataFrame(
        {"P1": genotypes, "P2": ["T"] * len(genotypes)},
        index=adata.obs_names,
        dtype=object,
    )
    adata.obsm["genotype_proportion"] = pd.DataFrame(
        {"P1": np.linspace(0.5, 1, len(genotypes)), "P2": 1.0}, index=adata.obs_names
    )
    adata.obsm["genotype_counts"] = pd.DataFrame(
        {"P1": [2] * len(genotypes), "P2": 1}, index=adata.obs_names
    )
    return adata


def test_check_genotypes_requires_called_genotypes():
    """Data without `call_genotypes` output raises ValueError; genotyped data passes."""
    with pytest.raises(ValueError, match="call_genotypes"):
        pl._check_genotypes(gapfill())
    pl._check_genotypes(genotyped(["AC"] * 8))


class TestPlotLogo:
    def test_raw_gapfill_frequencies_without_genotypes(self, capsys):
        """Without genotypes, the logo shows UMIs per base at each gapfill position."""
        logo, ax = pl.plot_logo(gapfill(), "P1", align=False)
        assert ax is logo.ax
        assert ax.get_ylabel() == "UMIs with Gapfill"
        assert ax.get_xlabel() == "Relative Position (bp)"
        assert list(logo.df.columns) == ["A", "C", "G", "T"]
        assert len(logo.df) == 3

    def test_cell_counts_per_genotype(self):
        """With genotypes, cells are counted per base; both alleles of a heterozygous call count and missing calls are skipped."""
        logo, ax = pl.plot_logo(
            genotyped(["AC"] * 6 + ["AG/AC", np.nan]), "P1", align=False
        )
        assert ax.get_ylabel() == "Cells with Genotype"
        assert (
            logo.df.loc[0, "A"] == 8
            and logo.df.loc[1, "C"] == 7
            and logo.df.loc[1, "G"] == 1
        )

    def test_reverse_complement_reverses_positions_and_complements_bases(self):
        """`reverse_complement_gapfill` reverses positions and complements bases."""
        logo, _ = pl.plot_logo(
            genotyped(["AC"] * 8), "P1", align=False, reverse_complement_gapfill=True
        )
        assert logo.df.loc[0, "G"] == 8 and logo.df.loc[1, "T"] == 8

    def test_group_comparison_is_normalized_difference(self):
        """Comparing two groups plots the difference of their normalized base frequencies."""
        adata = genotyped(["AC"] * 4 + ["AG"] * 4)
        logo, ax = pl.plot_logo(
            adata, "P1", groupby="group", group="g1", compare_to="g2", align=False
        )
        assert ax.get_ylabel() == "Cells with Genotype (Normalized)"
        assert logo.df.loc[1, "C"] == pytest.approx(1.0) and logo.df.loc[
            1, "G"
        ] == pytest.approx(-1.0)
        assert logo.df.loc[0, "A"] == pytest.approx(0.0)

    def test_single_group_subset_with_alignment(self):
        """`group` restricts the logo to one group, and alignment handles genotypes of different lengths."""
        adata = genotyped(["AC", "ACT", "AC", "ACT"] + ["AG"] * 4)
        logo, ax = pl.plot_logo(adata, "P1", groupby="group", group="g1", align=True)
        assert len(logo.df) == 3 and logo.df.loc[0, "A"] == 4

    def test_unknown_grouping_column_raises(self):
        """An unknown `groupby` column raises ValueError."""
        with pytest.raises(ValueError, match="Groupby"):
            pl.plot_logo(gapfill(), "P1", groupby="missing", group="x")


@pytest.mark.parametrize("plot", [pl.dotplot, pl.tracksplot, pl.matrixplot])
def test_grouped_scanpy_wrappers_reject_unknown_probes(plot):
    """`dotplot`, `tracksplot` and `matrixplot` reject unknown probes with ValueError."""
    with pytest.raises(ValueError, match="not found"):
        plot(gapfill(), "NOPE", "group", show=False)


@pytest.mark.known_defect
@pytest.mark.xfail(
    raises=ValueError,
    reason="`probe in Series` checks the index (probe|gapfill), so valid probes are rejected",
)
@pytest.mark.parametrize("plot", [pl.dotplot, pl.tracksplot, pl.matrixplot])
def test_grouped_scanpy_wrappers_plot_a_valid_probe(plot):
    """`dotplot`, `tracksplot` and `matrixplot` plot a probe that exists."""
    assert plot(gapfill(), "P1", "group", show=False) is not None


def test_violin_rejects_unknown_probe():
    """`violin` rejects an unknown probe with ValueError."""
    with pytest.raises(ValueError, match="not found"):
        pl.violin(gapfill(), "NOPE", show=False)


@pytest.mark.known_defect
@pytest.mark.xfail(
    raises=ValueError,
    reason="`probe in Series` checks the index (probe|gapfill), so valid probes are rejected",
)
def test_violin_plots_a_valid_probe():
    """`violin` plots a probe that exists."""
    assert pl.violin(gapfill(), "P1", show=False) is not None


class TestGenotypeEmbeddings:
    @pytest.fixture
    def adata(self):
        adata = genotyped(["AC"] * 4 + ["AG"] * 4)
        rng = np.random.default_rng(1)
        adata.obsm["X_umap"] = rng.normal(size=(8, 2))
        adata.obsm["X_tsne"] = rng.normal(size=(8, 2))
        return adata

    @pytest.mark.parametrize("plot", [pl.tsne, pl.umap])
    def test_embeddings_are_colored_by_genotype_and_clean_up_obs(self, adata, plot):
        """t-SNE and UMAP plots color by genotype and proportion, then remove the temporary `obs` columns."""
        axes = plot(adata, "P1", show=False)
        assert len(axes) == 2
        assert "genotype" not in adata.obs and "genotype_proportion" not in adata.obs

    @pytest.mark.parametrize("plot", [pl.tsne, pl.umap])
    def test_existing_obs_columns_trigger_a_warning(self, adata, plot, capsys):
        """Existing genotype columns in `obs` are overwritten with a warning."""
        adata.obs["genotype"] = "old"
        plot(adata, "P1", show=False)
        assert "Overwriting existing genotype" in capsys.readouterr().out

    def test_clustermap(self, adata, capsys):
        """The clustermap is drawn, warns before overwriting genotype columns and removes them afterwards."""
        adata.obs["genotype"] = "old"
        grid = pl.clustermap(adata, "P1", show=False)
        assert grid is not None and "genotype" not in adata.obs
        assert "Overwriting existing genotype column" in capsys.readouterr().out

    @pytest.mark.parametrize("plot", [pl.clustermap, pl.tsne, pl.umap])
    def test_unknown_genotype_or_missing_calls_raise(self, adata, plot):
        """Unknown probes or data without genotype calls raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            plot(adata, "P9", show=False)
        with pytest.raises(ValueError, match="call_genotypes"):
            plot(gapfill(), "P1", show=False)
