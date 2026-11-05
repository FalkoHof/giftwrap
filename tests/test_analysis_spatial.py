import sys
import types
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import spatialdata as sd
from shapely.geometry import Point
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Identity

import giftwrap.analysis.spatial as gsp
import giftwrap.analysis.tools as tl
import support
from support import to_dense


def hd_names(resolution, coords):
    return [f"s_{resolution:03d}um_{r:05d}_{c:05d}-1" for r, c in coords]


GRID = [(r, c) for r in range(4) for c in range(4)]


def spatial_gapfill(coords=GRID, sparse=True, X=None):
    X = (
        np.column_stack([np.ones(len(coords)), [r * 4 + c for r, c in coords]])
        if X is None
        else X
    )
    return support.make_gapfill_adata(
        X,
        ["P1", "P1"],
        ["A", "C"],
        sparse=sparse,
        obs_names=hd_names(2, coords),
        obs={"array_row": [r for r, _ in coords], "array_col": [c for _, c in coords]},
        extra_layers={"X_pcr_threshold_1": X, "total_counts": X, "other": X},
    )


class TestGuards:
    def test_spatial_coordinates_are_required(self):
        """Data without array coordinates is rejected as non-Visium HD, while spatial data passes."""
        with pytest.raises(ValueError, match="Visium HD"):
            gsp.assert_spatial(ad.AnnData(np.ones((1, 1))))
        gsp.assert_spatial(spatial_gapfill())

    def test_missing_optional_dependencies_raise_import_errors(self, monkeypatch):
        """Missing spatialdata, squidpy or spatialdata_plot raises ImportError naming the package."""
        gsp.check_squidpy()
        gsp.check_plotting()
        monkeypatch.setattr(gsp, "sd", None)
        monkeypatch.setattr(gsp, "sq", None)
        monkeypatch.setattr(gsp, "spatialdata_plot", None)
        with pytest.raises(ImportError, match="spatialdata"):
            gsp.assert_spatial(spatial_gapfill())
        with pytest.raises(ImportError, match="squidpy"):
            gsp.check_squidpy()
        with pytest.raises(ImportError, match="spatialdata_plot"):
            gsp.check_plotting()


class TestBin:
    @pytest.mark.parametrize("sparse", [True, False])
    def test_aggregates_counts_and_selected_layers_into_coarser_bins(self, sparse):
        """Bins get new names and coordinates; counts and count layers are summed and var, uns and sparsity are kept."""
        adata = spatial_gapfill(sparse=sparse)
        adata.uns["note"] = "kept"
        out = gsp.bin(adata, resolution=4)
        assert out.obs_names.tolist() == [
            "s_004um_00000_00000-1",
            "s_004um_00000_00001-1",
            "s_004um_00001_00000-1",
            "s_004um_00001_00001-1",
        ]
        assert out.obs["array_row"].tolist() == [0, 0, 1, 1] and out.obs[
            "array_col"
        ].tolist() == [0, 1, 0, 1]
        np.testing.assert_array_equal(
            to_dense(out.X), [[4, 10], [4, 18], [4, 42], [4, 50]]
        )
        assert set(out.layers.keys()) == {"X_pcr_threshold_1", "total_counts"}
        np.testing.assert_array_equal(
            to_dense(out.layers["total_counts"]), to_dense(out.X)
        )
        assert (
            out.var_names.tolist() == adata.var_names.tolist()
            and out.uns["note"] == "kept"
        )
        assert sp.issparse(out.X) == sparse

    def test_two_micron_resolution_is_returned_unchanged(self):
        """Binning at the native 2-micron resolution returns the same object."""
        adata = spatial_gapfill()
        assert gsp.bin(adata, 2) is adata

    def test_odd_resolution_and_non_spatial_data_are_rejected(self):
        """Odd resolutions raise AssertionError, and non-spatial data raises ValueError."""
        with pytest.raises(AssertionError, match="power of 2"):
            gsp.bin(spatial_gapfill(), 3)
        with pytest.raises(ValueError):
            gsp.bin(ad.AnnData(np.ones((1, 1))), 8)

    def test_genotypes_are_recalled_with_previous_arguments(self, capsys):
        """Binned data is genotyped again with the original call arguments."""
        adata = tl.call_genotypes(spatial_gapfill(), threshold=0.9)
        out = gsp.bin(adata, 8)
        assert "Calling genotypes for the binned data" in capsys.readouterr().out
        assert out.uns["genotype_call_args"]["threshold"] == 0.9
        assert out.obsm["genotype"].loc["s_008um_00000_00000-1", "P1"] in {"A/C", "C/A"}

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="bin count per row is floored, so distinct bins collide when the grid is not a multiple"
    )
    def test_bins_stay_distinct_for_grids_that_are_not_bin_multiples(self):
        """Bins stay distinct when the grid is not a multiple of the bin size."""
        out = gsp.bin(spatial_gapfill(coords=[(0, 4), (4, 0)]), 8)
        assert out.n_obs == 2

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="the 'power of 2' check only tests evenness, so e.g. 6um is accepted"
    )
    def test_non_power_of_two_resolution_is_rejected(self):
        """Resolutions that are not powers of two are rejected."""
        with pytest.raises(AssertionError):
            gsp.bin(spatial_gapfill(), 6)


def _wta_table(resolution, coords, dataset="ds"):
    names = hd_names(resolution, coords)
    region = f"{dataset}_square_{resolution:03d}um"
    table = ad.AnnData(
        sp.csr_matrix(np.ones((len(names), 2))),
        obs=pd.DataFrame(
            {"region": pd.Categorical([region] * len(names)), "location_id": names},
            index=names,
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    table.obsm["spatial"] = np.array([[c, r] for r, c in coords], dtype=float)
    return TableModel.parse(
        table, region=region, region_key="region", instance_key="location_id"
    )


def tiny_spatialdata(dataset="ds"):
    transform = {"": Identity()}
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            geometry=[Point(c, r) for r, c in GRID], index=hd_names(2, GRID)
        ).assign(radius=0.5),
        transformations=transform,
    )
    image = Image2DModel.parse(
        np.zeros((3, 8, 8)), dims=("c", "y", "x"), transformations=transform
    )
    return sd.SpatialData(
        images={f"{dataset}_hires_image": image},
        shapes={f"{dataset}_square_002um": shapes},
        tables={
            "square_002um": _wta_table(2, GRID),
            "square_008um": _wta_table(8, [(0, 0), (0, 1)]),
        },
    )


class TestJoinWithWta:
    def test_adds_gapfill_tables_aligned_to_wta_bins(self):
        """A gapfill table per WTA resolution is aligned to its bins, coordinates and region; bins without gapfill data are zero."""
        gf = spatial_gapfill(coords=GRID[:10])
        out = gsp.join_with_wta(tiny_spatialdata(), gf)
        assert set(out.tables.keys()) == {
            "square_002um",
            "square_008um",
            "gf_square_002um",
            "gf_square_008um",
        }
        t2 = out.tables["gf_square_002um"]
        assert t2.obs_names.tolist() == hd_names(2, GRID)
        np.testing.assert_array_equal(to_dense(t2.X)[:10], to_dense(gf.X))
        assert not to_dense(t2.X)[10:].any()
        assert t2.obs["array_row"].tolist() == [r for r, _ in GRID]
        assert t2.obs["location_id"].tolist() == hd_names(2, GRID)
        np.testing.assert_array_equal(
            t2.obsm["spatial"], out.tables["square_002um"].obsm["spatial"]
        )
        assert t2.uns[sd.models.TableModel.ATTRS_KEY]["region"] == "ds_square_002um"
        t8 = out.tables["gf_square_008um"]
        assert t8.obs_names.tolist() == [
            "s_008um_00000_00000-1",
            "s_008um_00000_00001-1",
        ]
        np.testing.assert_array_equal(to_dense(t8.X), [[10, 45], [0, 0]])

    def test_dense_inputs_with_extra_metadata_are_padded(self):
        """Dense counts and extra obs and obsm entries are padded with NaN for bins without gapfill data."""
        gf = spatial_gapfill(coords=GRID[:10], sparse=False)
        gf.obs["score"] = np.arange(10, dtype=float)
        gf.obs["label"] = ["x"] * 10
        gf.obsm["embedding"] = np.ones((10, 2))
        out = gsp.join_with_wta(tiny_spatialdata(), gf).tables["gf_square_002um"]
        assert np.isnan(out.obs["score"].iloc[-1]) and out.obsm["embedding"].shape == (
            16,
            2,
        )
        assert np.isnan(out.obsm["embedding"][-1]).all()

    def test_requires_spatial_gapfill_data(self):
        """Non-spatial gapfill data raises ValueError."""
        with pytest.raises(ValueError):
            gsp.join_with_wta(tiny_spatialdata(), ad.AnnData(np.ones((1, 1))))


class TestPlotGenotypes:
    def _joined(self, genotypes):
        gf = spatial_gapfill()
        out = gsp.join_with_wta(tiny_spatialdata(), gf)
        table = out.tables["gf_square_002um"]
        table.obsm["genotype"] = pd.DataFrame(
            {"P1": pd.Categorical(genotypes)}, index=table.obs_names
        )
        return out

    @pytest.mark.slow
    def test_spatialdata_plot_renders_image_and_bins(self):
        """SpatialData input is drawn over its image with labelled axes, and the temporary obs column is removed."""
        sdata = self._joined(["A", "C"] * 8)
        ax = gsp.plot_genotypes(sdata, "P1", dataset_id="ds")
        assert isinstance(ax, plt.Axes)
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Spatial 1", "Spatial 2")
        assert "giftwrap_genotype" not in sdata["square_002um"].obs

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=TypeError,
        reason="bins without a genotype (NaN) break spatialdata_plot colouring",
    )
    def test_spatialdata_plot_with_missing_genotypes(self):
        """Bins without a genotype are drawn without errors."""
        sdata = self._joined(["A", np.nan] * 8)
        gsp.plot_genotypes(sdata, "P1", dataset_id="ds")

    def test_anndata_fallback_rasterizes_genotypes(self):
        """AnnData input is rasterized with one colour per genotype, faded N/A bins and transparent empty pixels."""
        adata = spatial_gapfill(coords=[(0, 0), (1, 2), (2, 1)])
        adata.obsm["genotype"] = pd.DataFrame(
            {"P1": ["A", "C", np.nan]}, index=adata.obs_names, dtype=object
        )
        ax = gsp.plot_genotypes(adata, "P1")
        image = ax.images[0].get_array()
        assert image.shape == (3, 3, 4)
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend[0] == "N/A" and sorted(legend[1:]) == ["A", "C"]
        assert image[1, 2, 3] == pytest.approx(0.3)
        assert image[0, 0, 3] == 1.0 and image[2, 1, 3] == 1.0
        assert image[0, 1, 3] == 0.0

    def test_anndata_fallback_requires_coordinates(self):
        """AnnData without spatial coordinates raises ValueError."""
        adata = ad.AnnData(np.ones((1, 1)))
        adata.obsm["genotype"] = pd.DataFrame({"P1": ["A"]}, index=adata.obs_names)
        with pytest.raises(ValueError):
            gsp.plot_genotypes(adata, "P1")

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=TypeError,
        reason="_downsample_image mixes `>` and `|` without parentheses",
    )
    def test_anndata_fallback_at_coarser_resolution(self):
        """A coarser `resolution` downsamples the raster."""
        adata = spatial_gapfill()
        adata.obsm["genotype"] = pd.DataFrame({"P1": ["A"] * 16}, index=adata.obs_names)
        assert gsp.plot_genotypes(adata, "P1", resolution=4).images[
            0
        ].get_array().shape == (2, 2, 4)


class TestDownsampleImage:
    def test_rejects_non_square_or_non_rgba_images(self):
        """Non-square or non-RGBA images raise AssertionError."""
        with pytest.raises(AssertionError, match="NxN"):
            gsp._downsample_image(np.zeros((4, 2, 4)))
        with pytest.raises(AssertionError, match="4 channels"):
            gsp._downsample_image(np.zeros((4, 4, 3)))

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=TypeError,
        reason="`blocks > 0.5 | mask` is parsed as `blocks > (0.5 | mask)`",
    )
    def test_blocks_are_averaged_over_opaque_pixels(self):
        """Each block averages the colours of its opaque pixels, and empty blocks stay transparent."""
        image = np.zeros((4, 4, 4), dtype=np.float32)
        image[0, 0] = [1, 0, 0, 1]
        image[0, 1] = [0, 0, 1, 1]
        out = gsp._downsample_image(image, factor=2)
        assert out.shape == (2, 2, 4)
        np.testing.assert_allclose(out[0, 0], [0.5, 0, 0.5, 1])
        assert not out[1, 1].any()


class TestSpatialImputeWrapper:
    def _sdata(self, *names, cluster=True):
        tables = {}
        for name in names:
            table = ad.AnnData(np.zeros((2, 1)))
            if cluster:
                table.obs["cl"] = ["a", "b"]
            tables[name] = table
        return SimpleNamespace(tables=tables)

    def test_forwards_only_explicit_arguments(self, monkeypatch, capsys):
        """Only explicitly set options are forwarded, and the table is replaced by the result."""
        calls = []
        monkeypatch.setattr(
            gsp.tl,
            "impute_genotypes",
            lambda table, key, **kw: calls.append((key, kw)) or "imputed",
        )
        sdata = gsp.impute_genotypes(
            self._sdata("square_008um"), "cl", "square_008um", k=5, cores=2
        )
        assert calls == [("cl", {"k": 5, "cores": 2})]
        assert sdata.tables["square_008um"] == "imputed"
        gsp.impute_genotypes(
            self._sdata("square_008um"),
            "cl",
            "square_008um",
            threshold=0.7,
            impute_all=True,
            hold_out=0.1,
        )
        assert calls[-1] == (
            "cl",
            {"threshold": 0.7, "impute_all": True, "hold_out": 0.1},
        )
        assert (
            "Imputing genotypes for resolution square_008um" in capsys.readouterr().out
        )

    def test_without_resolution_every_present_resolution_is_imputed(self, monkeypatch):
        """Without a resolution, every `square_*` table is imputed from finest to coarsest."""
        seen = []
        monkeypatch.setattr(
            gsp.tl,
            "impute_genotypes",
            lambda table, key, **kw: seen.append(table) or table,
        )
        sdata = self._sdata("square_016um", "square_002um", "other")
        gsp.impute_genotypes(sdata, "cl")
        assert seen == [sdata.tables["square_002um"], sdata.tables["square_016um"]]

    def test_unknown_resolution_or_cluster_key_raise(self):
        """Unknown resolutions or cluster keys raise ValueError."""
        with pytest.raises(ValueError, match="not found in SpatialData"):
            gsp.impute_genotypes(self._sdata("square_002um"), "cl", "square_008um")
        with pytest.raises(ValueError, match="Cluster key 'cl' not found"):
            gsp.impute_genotypes(
                self._sdata("square_002um", cluster=False), "cl", "square_002um"
            )


@pytest.fixture
def stub_spatialdata_io(monkeypatch):
    """spatialdata_io is not a declared dependency; provide to_legacy_anndata by returning the table itself."""
    calls = []

    def to_legacy_anndata(sdata, table_name, include_images, coordinate_system):
        calls.append((table_name, include_images, coordinate_system))
        return sdata[table_name].copy()

    package = types.ModuleType("spatialdata_io")
    experimental = types.ModuleType("spatialdata_io.experimental")
    experimental.to_legacy_anndata = to_legacy_anndata
    package.experimental = experimental
    monkeypatch.setitem(sys.modules, "spatialdata_io", package)
    monkeypatch.setitem(sys.modules, "spatialdata_io.experimental", experimental)
    return calls


def _expression_table(n_side=6, n_genes=20):
    rng = np.random.default_rng(0)
    coords = [(r, c) for r in range(n_side) for c in range(n_side)]
    table = ad.AnnData(
        rng.poisson(3, size=(len(coords), n_genes)).astype(float),
        obs=pd.DataFrame(index=hd_names(2, coords)),
        var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
    )
    table.obsm["spatial"] = np.array([[c, r] for r, c in coords], dtype=float)
    return {"square_002um": table}


class TestCoclusteringRecipe:
    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=KeyError,
        reason="reads obsp['expr_connectivities'] but scanpy stores '<key>_connectivities'",
    )
    def test_combines_expression_and_spatial_graphs_and_stores_clusters(
        self, stub_spatialdata_io
    ):
        """Expression and spatial graphs are combined by weight, and the resulting clusters are stored."""
        sdata = _expression_table()
        out = gsp.recipe_spatial_expression_coclustering(
            sdata,
            n_neighbors=4,
            combination_weight=0.25,
            n_highly_variable_genes=10,
            coordinate_system="global",
        )
        table = out["square_002um"]
        assert stub_spatialdata_io == [("square_002um", False, "global")]
        assert table.obs["spatio_expression_cluster"].notna().all()
        assert table.obsp["combined_connectivities"].shape == (36, 36)
        assert "spatio_expression_cluster" in table.uns

    def test_requires_squidpy(self, stub_spatialdata_io, monkeypatch):
        """Without squidpy, the recipe raises ImportError."""
        monkeypatch.setattr(gsp, "sq", None)
        with pytest.raises(ImportError, match="squidpy"):
            gsp.recipe_spatial_expression_coclustering(_expression_table())

    def test_neighbor_graphs_are_computed_on_the_legacy_table(
        self, stub_spatialdata_io, monkeypatch
    ):
        """Expression and grid-based spatial neighbor graphs are built on the legacy table with the given sizes."""
        seen = {}
        monkeypatch.setattr(
            gsp.sc.pp, "neighbors", lambda adata, **kw: seen.setdefault("expr", kw)
        )
        monkeypatch.setattr(
            gsp.sq.gr,
            "spatial_neighbors",
            lambda adata, **kw: seen.setdefault("spatial", kw),
        )
        with pytest.raises(KeyError):
            gsp.recipe_spatial_expression_coclustering(
                _expression_table(), n_neighbors=4, n_spatial_neighbors=6
            )
        assert seen["expr"] == {
            "n_neighbors": 4,
            "use_rep": "X_pca",
            "key_added": "expr_connectivities",
        }
        assert seen["spatial"] == {
            "coord_type": "grid",
            "n_neighs": 6,
            "key_added": "spatial_connectivities",
        }
        assert stub_spatialdata_io == [("square_002um", False, None)]

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=ModuleNotFoundError,
        reason="imports spatialdata_io, which no extra declares",
    )
    def test_runs_with_the_declared_spatial_dependencies(self):
        """The recipe runs with only the declared spatial extras installed."""
        gsp.recipe_spatial_expression_coclustering(_expression_table(), n_neighbors=4)
