import itertools
import operator
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import support
from giftwrap.analysis_utils import (
    DummyResult,
    ItertoolsWrapper,
    iter_layers,
    maybe_multiprocess,
    parse_array_coords,
    read_h5_file,
    read_sparse_matrix,
    repair_array_axis,
)
from giftwrap.pipeline_utils import write_sparse_matrix

SAMPLE_COUNTS = Path(__file__).parents[1] / "sample_data" / "counts.1.filtered.h5"


class TestDummyResult:
    def test_get_returns_wrapped_value_ignoring_arguments(self):
        """`get()` returns the wrapped result regardless of timeout arguments."""
        result = DummyResult([1, 2])
        assert result.get() == [1, 2]
        assert result.get(timeout=5) == [1, 2]

    def test_status_methods_report_completion(self):
        """`wait`, `ready` and `successful` behave like an already finished AsyncResult."""
        result = DummyResult(None)
        assert result.wait() is None
        assert result.ready() is True
        assert result.successful() is True


class TestItertoolsWrapper:
    def test_starmap_is_lazy_itertools_starmap(self):
        """`starmap` returns a lazy `itertools.starmap`, unlike `Pool.starmap`."""
        out = ItertoolsWrapper().starmap(operator.mul, [(2, 3), (4, 5)])
        assert isinstance(out, itertools.starmap)
        assert list(out) == [6, 20]

    def test_starmap_async_wraps_results_in_dummy_result(self):
        """`starmap_async` returns a ready DummyResult wrapping the results."""
        job = ItertoolsWrapper().starmap_async(operator.add, [(1, 1), (2, 3)])
        assert isinstance(job, DummyResult)
        assert job.ready()
        assert list(job.get()) == [2, 5]


class TestMaybeMultiprocess:
    @pytest.mark.parametrize("cores", [-1, 0, 1])
    def test_single_core_yields_inline_wrapper(self, cores):
        """One core or fewer runs jobs inline through an ItertoolsWrapper."""
        with maybe_multiprocess(cores) as pool:
            assert isinstance(pool, ItertoolsWrapper)
            assert list(pool.starmap(operator.mul, [(3, 3)])) == [9]

    @pytest.mark.multiprocess
    def test_multiple_cores_use_a_real_process_pool(self):
        """Two or more cores use a real process pool that returns a list."""
        with maybe_multiprocess(2) as pool:
            assert not isinstance(pool, ItertoolsWrapper)
            assert pool.starmap(operator.mul, [(2, 3), (4, 5)]) == [6, 20]


class TestReadSparseMatrix:
    def test_round_trips_matrix_written_by_write_sparse_matrix(self, tmp_path):
        """A matrix written by `write_sparse_matrix` reads back as the same CSR matrix."""
        mat = sp.csr_matrix(np.array([[1, 0, 2], [0, 0, 3]], dtype=np.uint32))
        with h5py.File(tmp_path / "m.h5", "w") as f:
            write_sparse_matrix(f, "m", mat)
        with h5py.File(tmp_path / "m.h5", "r") as f:
            out = read_sparse_matrix(f, "m")
        assert sp.isspmatrix_csr(out)
        assert out.shape == (2, 3)
        np.testing.assert_array_equal(out.toarray(), mat.toarray())

    def test_empty_matrix_keeps_shape(self, tmp_path):
        """An all-zero matrix keeps its shape and has no stored entries."""
        with h5py.File(tmp_path / "m.h5", "w") as f:
            write_sparse_matrix(f, "m", sp.csr_matrix((4, 2)))
        with h5py.File(tmp_path / "m.h5", "r") as f:
            out = read_sparse_matrix(f, "m")
        assert out.shape == (4, 2)
        assert out.nnz == 0


class TestIterLayers:
    def test_yields_named_layers_of_anndata(self):
        """All named layers are yielded, whether dense or sparse."""
        adata = ad.AnnData(
            np.ones((2, 2)),
            layers={"a": np.zeros((2, 2)), "b": sp.csr_matrix(np.eye(2))},
        )
        assert sorted(name for name, _ in iter_layers(adata)) == ["a", "b"]

    def test_skips_the_none_alias_for_x(self):
        """The `None` key that anndata>=0.13 uses to alias X is skipped."""
        fake = SimpleNamespace(layers={None: "X", "counts": "C"})
        assert list(iter_layers(fake)) == [("counts", "C")]

    def test_no_layers_yields_nothing(self):
        """An AnnData object without layers yields nothing."""
        assert list(iter_layers(ad.AnnData(np.ones((1, 1))))) == []


class TestParseArrayCoords:
    def test_parses_visium_hd_row_and_column(self):
        """Row and column are read from `s_<res>um_<row>_<col>-<plex>` barcodes."""
        rows, cols = parse_array_coords(
            ["s_002um_00010_00020-1", "s_008um_00003_00104-2"]
        )
        assert rows.tolist() == [10, 3]
        assert cols.tolist() == [20, 104]

    def test_preserves_series_index_for_alignment(self):
        """The input Series index is kept so results align with `adata.obs`."""
        barcodes = pd.Series(["s_002um_00001_00002-1"], index=["cellA"])
        rows, cols = parse_array_coords(barcodes)
        assert rows.index.tolist() == ["cellA"]
        assert cols.loc["cellA"] == 2

    def test_classic_visium_barcodes_yield_minus_one(self):
        """Classic Visium barcodes carry no coordinates and yield -1."""
        rows, cols = parse_array_coords(["ACGTACGTACGTACGT-1", "TTTTGGGGCCCCAAAA-1"])
        assert rows.tolist() == [-1, -1]
        assert cols.tolist() == [-1, -1]

    def test_mixed_and_non_numeric_parts_fall_back_to_minus_one(self):
        """Missing or non-numeric coordinate parts fall back to -1 per axis."""
        rows, cols = parse_array_coords(
            ["s_002um_00005_00006-1", "ACGT-1", "s_002um_XX_7-1"]
        )
        assert rows.tolist() == [5, -1, -1]
        assert cols.tolist() == [6, -1, 7]


class TestRepairArrayAxis:
    def test_integer_values_are_kept(self):
        """Valid integer coordinates are returned unchanged."""
        out = repair_array_axis(pd.Series([3, 4]), pd.Series([9, 9]))
        assert out.tolist() == [3, 4]
        assert out.dtype.kind == "i"

    def test_na_strings_are_replaced_by_parsed_coordinates(self):
        """Unparseable stored values such as "<NA>" take the barcode-derived coordinate."""
        out = repair_array_axis(pd.Series(["7", "<NA>", "abc"]), pd.Series([0, 11, 12]))
        assert out.tolist() == [7, 11, 12]

    def test_unrecoverable_values_become_minus_one(self):
        """Values that can be neither parsed nor recovered become -1."""
        out = repair_array_axis(pd.Series(["<NA>", None]), pd.Series([np.nan, -1]))
        assert out.tolist() == [-1, -1]


def _write_basic(path, **kwargs):
    return support.write_counts_h5(
        path,
        [[1, 0, 2], [0, 3, 0]],
        [("GENEA x", "ACG"), ("GENEA x", "AGG"), ("GENEB y", "T")],
        ["AAAA-1", "CCCC-1"],
        **kwargs,
    )


class TestReadH5File:
    def test_builds_anndata_with_matrices_annotations_and_run_metadata(self, tmp_path):
        """Matrices, cell and probe annotations, and run metadata are read from a collect-style file."""
        path = _write_basic(
            tmp_path / "counts.1.h5",
            total_reads=[[4, 0, 2], [0, 9, 0]],
            cell_index=[5, 7],
            probe_index=[0, 0, 1],
            cell_metadata={
                "barcode": ["AAAA-1", "CCCC-1"],
                "plex_id": [1, 1],
                "plex_seq": ["", "GG"],
            },
        )
        adata = read_h5_file(path)

        assert adata.shape == (2, 3)
        np.testing.assert_array_equal(adata.X.toarray(), [[1, 0, 2], [0, 3, 0]])
        np.testing.assert_array_equal(
            adata.layers["total_reads"].toarray(), [[4, 0, 2], [0, 9, 0]]
        )
        assert "percent_supporting" in adata.layers
        assert adata.obs_names.tolist() == ["AAAA-1", "CCCC-1"]
        assert adata.obs["cell_index"].tolist() == [5, 7]
        assert adata.obs["plex_id"].dtype.kind == "i"
        assert adata.obs["plex_seq"].tolist() == ["", "GG"]
        assert adata.var_names.tolist() == ["GENEA x|ACG", "GENEA x|AGG", "GENEB y|T"]
        assert adata.var["probe_index"].tolist() == [0, 0, 1]
        assert adata.var["expected_gapfill"].tolist() == ["ACG"] * 3
        assert adata.var["reference_gapfill"].tolist() == ["AGG"] * 3
        assert adata.var["gene"].tolist() == ["GENEA", "GENEA", "GENEB"]
        assert adata.uns["plex"] == "1"
        assert adata.uns["n_probe_gapfill_combinations"] == 3
        assert adata.uns["max_pcr_duplicates"] == -1
        assert adata.uns["probe_metadata"]["probe"].tolist() == ["GENEA x", "GENEB y"]

    def test_optional_indices_and_gene_column_may_be_absent(self, tmp_path):
        """Files without cell or probe indices or gene names still load."""
        meta = pd.DataFrame(
            {
                "name": ["GENEA x", "GENEB y"],
                "lhs_probe": ["A", "C"],
                "rhs_probe": ["G", "T"],
                "gap_probe_sequence": ["ACG", "T"],
                "original_gap_probe_sequence": ["AGG", "C"],
                "index": [0, 1],
            }
        )
        adata = read_h5_file(_write_basic(tmp_path / "c.h5", probe_metadata=meta))
        assert "cell_index" not in adata.obs
        assert "probe_index" not in adata.var
        assert "gene" not in adata.var
        assert adata.var["reference_gapfill"].tolist() == ["AGG", "AGG", "C"]

    def test_duplicate_probe_names_raise(self, tmp_path):
        """Duplicate probe names in the probe metadata raise ValueError."""
        meta = pd.DataFrame(
            {
                "name": ["GENEA x", "GENEA x"],
                "lhs_probe": ["A", "C"],
                "rhs_probe": ["G", "T"],
                "gap_probe_sequence": ["ACG", "T"],
                "original_gap_probe_sequence": ["AGG", "C"],
                "index": [0, 1],
            }
        )
        with pytest.raises(ValueError, match="not unique"):
            read_h5_file(_write_basic(tmp_path / "c.h5", probe_metadata=meta))

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="cell metadata is cast with astype(int) first, silently truncating float columns"
    )
    def test_float_cell_metadata_is_preserved(self, tmp_path):
        """Float cell-metadata columns keep their fractional values."""
        path = _write_basic(
            tmp_path / "c.h5",
            cell_metadata={"barcode": ["AAAA-1", "CCCC-1"], "score": [0.25, 1.5]},
        )
        assert read_h5_file(path).obs["score"].tolist() == [0.25, 1.5]

    def test_pcr_threshold_layers_are_loaded(self, tmp_path):
        """PCR-threshold matrices become `X_pcr_threshold_<n>` layers, with the maximum in uns."""
        layers = {
            1: [[1, 0, 2], [0, 3, 0]],
            2: [[1, 0, 1], [0, 2, 0]],
            3: [[0, 0, 1], [0, 1, 0]],
        }
        adata = read_h5_file(_write_basic(tmp_path / "c.h5", pcr_layers=layers))
        assert adata.uns["max_pcr_duplicates"] == 3
        np.testing.assert_array_equal(
            adata.layers["X_pcr_threshold_1"].toarray(), layers[1]
        )
        np.testing.assert_array_equal(
            adata.layers["X_pcr_threshold_2"].toarray(), layers[2]
        )

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="read_h5_file loads pcr1..pcr(N-1); the highest written threshold layer is dropped"
    )
    def test_highest_pcr_threshold_layer_is_loaded(self, tmp_path):
        """The highest written PCR-threshold layer is loaded as well."""
        layers = {
            1: [[1, 0, 2], [0, 3, 0]],
            2: [[1, 0, 1], [0, 2, 0]],
            3: [[0, 0, 1], [0, 1, 0]],
        }
        adata = read_h5_file(_write_basic(tmp_path / "c.h5", pcr_layers=layers))
        assert "X_pcr_threshold_3" in adata.layers

    def test_integer_spatial_coordinates_are_kept_without_repair(
        self, tmp_path, capsys
    ):
        """Integer array coordinates are kept and no repair warning is printed."""
        barcodes = ["s_002um_00010_00020-1", "s_002um_00030_00040-1"]
        path = support.write_counts_h5(
            tmp_path / "c.h5",
            [[1], [2]],
            [("GENEA x", "A")],
            barcodes,
            cell_metadata={
                "barcode": barcodes,
                "array_row": [10, 30],
                "array_col": [20, 40],
            },
        )
        adata = read_h5_file(path)
        assert adata.obs["array_row"].tolist() == [10, 30]
        assert adata.obs["array_col"].tolist() == [20, 40]
        assert "Regenerating" not in capsys.readouterr().out

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=TypeError,
        reason="read_h5_file passes a mask as a 3rd argument to the 2-argument repair_array_axis",
    )
    def test_legacy_na_string_coordinates_are_repaired_from_barcodes(self, tmp_path):
        """Legacy "<NA>" coordinates are rebuilt from the spatial barcodes."""
        barcodes = ["s_002um_00010_00020-1", "s_002um_00030_00040-1"]
        path = support.write_counts_h5(
            tmp_path / "c.h5",
            [[1], [2]],
            [("GENEA x", "A")],
            barcodes,
            cell_metadata={
                "barcode": barcodes,
                "array_row": ["10", "<NA>"],
                "array_col": ["<NA>", "40"],
            },
        )
        adata = read_h5_file(path)
        assert adata.obs["array_row"].tolist() == [10, 30]
        assert adata.obs["array_col"].tolist() == [20, 40]

    def test_reads_bundled_sample_counts_file(self):
        """The bundled sample counts file loads with shape and metadata that agree."""
        adata = read_h5_file(SAMPLE_COUNTS)
        assert adata.shape == (28011, 6042)
        assert adata.uns["n_cells"] == adata.n_obs
        assert adata.uns["n_probe_gapfill_combinations"] == adata.n_vars
        assert sp.issparse(adata.X)
        assert set(adata.layers.keys()) >= {"total_reads", "percent_supporting"}
        assert all("|" in name for name in adata.var_names[:20])
        assert adata.obs_names.is_unique
