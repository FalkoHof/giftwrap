import h5py
import numpy as np
import pytest
import scipy.sparse as sp

import support
from giftwrap.analysis_utils import read_h5_file, read_sparse_matrix
from giftwrap.pipeline_utils import (
    filter_h5_file_by_barcodes,
    filter_h5_file_by_pcr_dups,
    real_gapfill_mask,
)

PROBES = [("GENEA x", "ACG"), ("GENEB y", "T")]
STEP3_HEADER = [
    "cell_idx",
    "probe_idx",
    "probe_barcode",
    "umi",
    "gapfill",
    "pcr_duplicate_count",
    "percent_supporting",
]


def _matrix(path, layer="data", group="matrix"):
    with h5py.File(path, "r") as f:
        return read_sparse_matrix(f[group], layer).toarray()


def _barcodes(path):
    with h5py.File(path, "r") as f:
        return [b.decode() for b in f["matrix"]["barcode"][:]]


class TestFilterByBarcodes:
    def _write(self, path, barcodes, **kwargs):
        X = [[1, 0], [0, 2], [3, 4]]
        return support.write_counts_h5(
            path, X, PROBES, barcodes, cell_index=[10, 11, 12], **kwargs
        )

    def test_exact_matches_keep_requested_rows(self, tmp_path, capsys):
        """Exact barcode matches keep only the requested rows of every matrix and index."""
        src = self._write(tmp_path / "in.h5", ["AAAA-1", "CCCC-1", "GGGG-1"])
        filter_h5_file_by_barcodes(src, tmp_path / "out.h5", ["GGGG-1", "AAAA-1"])
        assert _barcodes(tmp_path / "out.h5") == ["AAAA-1", "GGGG-1"]
        np.testing.assert_array_equal(_matrix(tmp_path / "out.h5"), [[1, 0], [3, 4]])
        np.testing.assert_array_equal(
            _matrix(tmp_path / "out.h5", "total_reads"), [[1, 0], [3, 4]]
        )
        with h5py.File(tmp_path / "out.h5", "r") as f:
            assert f["matrix"]["cell_index"][:].tolist() == [10, 12]
            assert f.attrs["n_cells"] == 2
        assert "Padding" not in capsys.readouterr().out

    def test_embedded_plex_sequence_is_trimmed_to_match_cellranger_barcodes(
        self, tmp_path
    ):
        """Barcodes with an embedded plex sequence match plain CellRanger barcodes."""
        src = self._write(
            tmp_path / "in.h5", ["AAAACCCCGG-1", "CCCCAAAAGG-1", "GGGGTTTTGG-1"]
        )
        filter_h5_file_by_barcodes(
            src, tmp_path / "out.h5", ["CCCCAAAA-1"], pad_matrix=False
        )
        assert _barcodes(tmp_path / "out.h5") == ["CCCCAAAAGG-1"]

    def test_gem_suffix_is_ignored_as_last_resort(self, tmp_path):
        """When nothing else matches, barcodes are compared without their -N suffix."""
        src = self._write(tmp_path / "in.h5", ["AAAA-2", "CCCC-2", "GGGG-2"])
        filter_h5_file_by_barcodes(
            src, tmp_path / "out.h5", ["CCCC-1"], pad_matrix=False
        )
        assert _barcodes(tmp_path / "out.h5") == ["CCCC-2"]

    def test_requesting_every_barcode_leaves_an_unchanged_copy(self, tmp_path):
        """Requesting every barcode leaves the copied file byte-identical."""
        src = self._write(tmp_path / "in.h5", ["AAAA-1", "CCCC-1", "GGGG-1"])
        filter_h5_file_by_barcodes(
            src, tmp_path / "out.h5", ["AAAA-1", "CCCC-1", "GGGG-1"]
        )
        assert (tmp_path / "out.h5").read_bytes() == src.read_bytes()

    def test_no_overlap_raises_with_examples(self, tmp_path):
        """No overlap raises ValueError quoting example file and requested barcodes."""
        src = self._write(tmp_path / "in.h5", ["AAAA-1", "CCCC-1", "GGGG-1"])
        with pytest.raises(ValueError, match="'AAAA-1'.*'TTTT-1'"):
            filter_h5_file_by_barcodes(src, tmp_path / "out.h5", ["TTTT-1"])
        with pytest.raises(ValueError, match="no barcodes requested"):
            filter_h5_file_by_barcodes(src, tmp_path / "out2.h5", [])

    def test_missing_requested_cells_are_padded_with_zero_rows_and_typed_metadata(
        self, tmp_path, capsys
    ):
        """Requested cells absent from the file get zero rows, cell index -1 and dtype-preserving metadata."""
        file_bcs = [
            "s_002um_00001_00002-1",
            "s_002um_00003_00004-1",
            "s_002um_00009_00009-1",
        ]
        src = self._write(
            tmp_path / "in.h5",
            file_bcs,
            cell_metadata={
                "barcode": file_bcs,
                "plex_id": [1, 1, 1],
                "plex_seq": ["", "", ""],
                "score": [0.5, 1.5, 2.5],
                "in_tissue": [1, 1, 1],
                "array_row": [1, 3, 9],
                "array_col": [2, 4, 9],
            },
        )
        requested = ["s_002um_00003_00004-1", "s_002um_00005_00006-1"]
        filter_h5_file_by_barcodes(src, tmp_path / "out.h5", requested)
        assert "Padding 1 unseen cells" in capsys.readouterr().out
        assert _barcodes(tmp_path / "out.h5") == requested
        np.testing.assert_array_equal(_matrix(tmp_path / "out.h5"), [[0, 2], [0, 0]])
        with h5py.File(tmp_path / "out.h5", "r") as f:
            assert f["matrix"]["cell_index"][:].tolist() == [11, -1]
        adata = read_h5_file(tmp_path / "out.h5")
        assert adata.obs["array_row"].tolist() == [3, 5]
        assert adata.obs["array_col"].tolist() == [4, 6]
        assert adata.obs["plex_id"].tolist() == [1, -1]
        assert adata.obs["in_tissue"].tolist() == [1, -1]
        assert adata.obs["plex_seq"].tolist() == ["", ""]
        with h5py.File(tmp_path / "out.h5", "r") as f:
            score = f["cell_metadata"]["score"][:]
        assert score[0] == 1.5 and np.isnan(score[1])

    def test_padding_can_be_disabled(self, tmp_path):
        """With `pad_matrix=False`, requested barcodes absent from the file are ignored."""
        src = self._write(tmp_path / "in.h5", ["AAAA-1", "CCCC-1", "GGGG-1"])
        filter_h5_file_by_barcodes(
            src, tmp_path / "out.h5", ["CCCC-1", "TTTT-1"], pad_matrix=False
        )
        assert _barcodes(tmp_path / "out.h5") == ["CCCC-1"]

    def test_metadata_without_barcode_column_is_filtered_by_position(self, tmp_path):
        """Cell metadata without a barcode column is filtered by row position."""
        src = self._write(
            tmp_path / "in.h5",
            ["AAAA-1", "CCCC-1", "GGGG-1"],
            cell_metadata={"plex_id": [1, 2, 3]},
        )
        filter_h5_file_by_barcodes(src, tmp_path / "out.h5", ["CCCC-1", "GGGG-1"])
        with h5py.File(tmp_path / "out.h5", "r") as f:
            assert f["cell_metadata"]["plex_id"][:].tolist() == [2, 3]
            assert [c.decode() for c in f["cell_metadata"]["columns"][:]] == ["plex_id"]

    def test_pcr_threshold_layers_below_the_maximum_are_filtered(self, tmp_path):
        """PCR-threshold layers are filtered and padded like the main matrices."""
        layers = {
            1: [[1, 0], [0, 2], [3, 4]],
            2: [[1, 0], [0, 1], [2, 4]],
            3: [[0, 0], [0, 1], [1, 1]],
        }
        src = self._write(
            tmp_path / "in.h5", ["AAAA-1", "CCCC-1", "GGGG-1"], pcr_layers=layers
        )
        filter_h5_file_by_barcodes(src, tmp_path / "out.h5", ["CCCC-1", "TTTT-1"])
        np.testing.assert_array_equal(
            _matrix(tmp_path / "out.h5", "pcr1", "pcr_thresholded_counts"),
            [[0, 2], [0, 0]],
        )
        np.testing.assert_array_equal(
            _matrix(tmp_path / "out.h5", "pcr2", "pcr_thresholded_counts"),
            [[0, 1], [0, 0]],
        )

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="only pcr1..pcr(N-1) are filtered; pcrN keeps the unfiltered rows"
    )
    def test_highest_pcr_threshold_layer_is_filtered_too(self, tmp_path):
        """The highest PCR-threshold layer is filtered like the others."""
        layers = {
            1: [[1, 0], [0, 2], [3, 4]],
            2: [[1, 0], [0, 1], [2, 4]],
            3: [[0, 0], [0, 1], [1, 1]],
        }
        src = self._write(
            tmp_path / "in.h5", ["AAAA-1", "CCCC-1", "GGGG-1"], pcr_layers=layers
        )
        filter_h5_file_by_barcodes(src, tmp_path / "out.h5", ["CCCC-1"])
        assert _matrix(tmp_path / "out.h5", "pcr3", "pcr_thresholded_counts").shape == (
            1,
            2,
        )


class TestFilterByPcrDuplicates:
    def _counts(
        self,
        path,
        X=((2, 1), (1, 0)),
        total=((6, 1), (4, 0)),
        probes=PROBES,
        probe_index=(0, 1),
    ):
        return support.write_counts_h5(
            path,
            X,
            probes,
            ["AAAA-1", "CCCC-1"],
            total_reads=total,
            cell_index=[0, 1],
            probe_index=list(probe_index),
        )

    def _reads(self, path, rows):
        return support.write_tsv(path, STEP3_HEADER, rows)

    def test_umis_below_threshold_are_subtracted_from_counts_and_reads(self, tmp_path):
        """UMIs with fewer reads than the threshold are removed from counts and read totals."""
        src = self._counts(tmp_path / "counts.1.h5")
        reads = self._reads(
            tmp_path / "probe_reads.tsv.gz",
            [
                [0, 0, 1, "U1", "ACG", 1, 1.0],
                [0, 0, 1, "U2", "ACG", 5, 1.0],
                [0, 1, 1, "U3", "T", 1, 1.0],
                [1, 0, 1, "U4", "ACG", 4, 1.0],
                [1, 0, 2, "U5", "ACG", 1, 1.0],
                [9, 0, 1, "U6", "ACG", 1, 1.0],
                ["short", "line"],
            ],
        )
        out = tmp_path / "counts.1.filtered.h5"
        filter_h5_file_by_pcr_dups(reads, src, out, reads_per_gapfill=2, probe_bc="1")
        np.testing.assert_array_equal(_matrix(out), [[1, 0], [1, 0]])
        np.testing.assert_array_equal(_matrix(out, "total_reads"), [[5, 0], [4, 0]])
        assert _matrix(out, "percent_supporting")[0, 1] == 0.0
        np.testing.assert_array_equal(_matrix(src), [[2, 1], [1, 0]])

    def test_counts_never_go_negative(self, tmp_path):
        """Subtracting more UMIs or reads than were stored stops at zero."""
        src = self._counts(
            tmp_path / "c.h5", X=((1, 0), (0, 0)), total=((1, 0), (0, 0))
        )
        reads = self._reads(
            tmp_path / "r.tsv",
            [[0, 0, 1, "U1", "ACG", 1, 1.0], [0, 0, 1, "U2", "ACG", 1, 1.0]],
        )
        filter_h5_file_by_pcr_dups(reads, src, tmp_path / "out.h5", 3, "1")
        assert _matrix(tmp_path / "out.h5")[0, 0] == 0
        assert _matrix(tmp_path / "out.h5", "total_reads")[0, 0] == 0

    def test_no_low_support_umis_leaves_counts_untouched(self, tmp_path, capsys):
        """When every UMI passes, the copy is unchanged and a message says so."""
        src = self._counts(tmp_path / "c.h5")
        reads = self._reads(tmp_path / "r.tsv", [[0, 0, 1, "U1", "ACG", 5, 1.0]])
        filter_h5_file_by_pcr_dups(reads, src, tmp_path / "out.h5", 2, "1")
        assert "no changes made" in capsys.readouterr().out
        assert (tmp_path / "out.h5").read_bytes() == src.read_bytes()

    def test_an_existing_output_file_is_filtered_in_place(self, tmp_path):
        """An existing output, e.g. a barcode-filtered file, is filtered in place, not replaced."""
        src = self._counts(tmp_path / "c.h5")
        out = self._counts(
            tmp_path / "out.h5", X=((5, 5), (5, 5)), total=((9, 9), (9, 9))
        )
        reads = self._reads(tmp_path / "r.tsv", [[1, 1, 1, "U1", "T", 1, 1.0]])
        filter_h5_file_by_pcr_dups(reads, src, out, 2, "1")
        np.testing.assert_array_equal(_matrix(out), [[5, 5], [5, 4]])

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="recomputed percent_supporting is written on a 0-100 scale while stored values are 0-1"
    )
    def test_recomputed_percent_supporting_stays_a_fraction(self, tmp_path):
        """Recomputed support values stay on the 0-1 scale of the stored values."""
        src = self._counts(tmp_path / "c.h5")
        reads = self._reads(
            tmp_path / "r.tsv",
            [[0, 0, 1, "U1", "ACG", 1, 1.0], [0, 0, 1, "U2", "ACG", 5, 1.0]],
        )
        filter_h5_file_by_pcr_dups(reads, src, tmp_path / "out.h5", 2, "1")
        assert 0.0 <= _matrix(tmp_path / "out.h5", "percent_supporting")[0, 0] <= 1.0

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="UMIs are mapped to columns by probe index only, ignoring the gapfill of each column"
    )
    def test_subtraction_targets_the_umi_gapfill_column(self, tmp_path):
        """Each UMI is subtracted from the column of its own gapfill."""
        probes = [("GENEA x", "ACG"), ("GENEA x", "AGG")]
        src = self._counts(
            tmp_path / "c.h5",
            X=((2, 1), (0, 0)),
            total=((6, 3), (0, 0)),
            probes=probes,
            probe_index=(0, 0),
        )
        reads = self._reads(
            tmp_path / "r.tsv",
            [
                [0, 0, 1, "U1", "ACG", 1, 1.0],
                [0, 0, 1, "U2", "ACG", 5, 1.0],
                [0, 0, 1, "U3", "AGG", 3, 1.0],
            ],
        )
        filter_h5_file_by_pcr_dups(reads, src, tmp_path / "out.h5", 2, "1")
        np.testing.assert_array_equal(_matrix(tmp_path / "out.h5"), [[1, 1], [0, 0]])


class TestRealGapfillMask:
    def test_excludes_zero_bp_control_probes(self):
        """Probes with "0bp" in their name are excluded."""
        adata = support.make_gapfill_adata(
            np.ones((1, 3)), ["GENEA x", "GENEB 0bp", "GENEC0bp"], ["A", "", ""]
        )
        assert real_gapfill_mask(adata).tolist() == [True, False, False]

    def test_without_probe_annotation_everything_is_real(self):
        """Without a `probe` column, every feature counts as a real gapfill."""
        import anndata as ad

        assert (
            real_gapfill_mask(ad.AnnData(sp.csr_matrix(np.ones((2, 4))))).tolist()
            == [True] * 4
        )
