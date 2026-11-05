import os
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

import giftwrap
import giftwrap.step4_collect_counts as step4
import support
from giftwrap.analysis_utils import read_h5_file, read_sparse_matrix
from giftwrap.pipeline_utils import read_barcodes, read_manifest, write_manifest

STEP3_HEADER = [
    "cell_idx",
    "probe_idx",
    "probe_barcode",
    "umi",
    "gapfill",
    "pcr_duplicate_count",
    "percent_supporting",
]
A, B = "GENEA c.1A>T", "GENEB c.2C>G"
READS = [
    [0, 0, 1, "U1", "TTG", 1, 1.0],
    [0, 0, 1, "U2", "TTG", 3, 0.5],
    [0, 1, 1, "U3", "GCAT", 2, 1.0],
    [1, 0, 1, "U4", "TAG", 1, 1.0],
    [1, 0, 2, "U5", "TAG", 9, 1.0],
    [7, 0, 1, "U6", "TTG", 1, 1.0],
]


def _output_dir(tmp_path, plexes=(1, 1, 1), reads=READS, name="run"):
    out = tmp_path / name
    out.mkdir()
    write_manifest(out, support.manifest_df(), {"technology": "Flex"})
    support.write_tsv(
        out / "barcodes.tsv.gz",
        ["barcode", "plex_id", "plex_seq"],
        [[f"{support.CELLS[i]}-1", p, "ACTTTAGG"] for i, p in enumerate(plexes)],
    )
    support.write_tsv(out / "probe_reads.tsv.gz", STEP3_HEADER, reads)
    return out


def _collect(out, **kwargs):
    args = (
        dict(
            overwrite=False,
            plex="1",
            multiplex=False,
            flatten=False,
            max_pcr_thresholds=10,
        )
        | kwargs
    )
    step4.collect_counts(
        out / "probe_reads.tsv.gz", out, read_manifest(out), read_barcodes(out), **args
    )
    return out / f"counts.{args['plex']}.h5"


def _layer(path, name, group="matrix"):
    with h5py.File(path, "r") as f:
        return read_sparse_matrix(f[group], name).toarray()


class TestCollectCounts:
    def test_builds_count_read_and_support_matrices(self, tmp_path):
        """UMI counts, read totals and mean support are built per cell and probe/gapfill."""
        path = _collect(_output_dir(tmp_path))
        np.testing.assert_array_equal(
            _layer(path, "data"), [[0, 2, 1], [1, 0, 0], [0, 0, 0]]
        )
        np.testing.assert_array_equal(
            _layer(path, "total_reads"), [[0, 4, 2], [1, 0, 0], [0, 0, 0]]
        )
        np.testing.assert_allclose(
            _layer(path, "percent_supporting"),
            [[0, 1.5 / 2.0001, 1 / 1.0001], [1 / 1.0001, 0, 0], [0, 0, 0]],
            rtol=1e-6,
        )

    def test_writes_index_metadata_and_attributes(self, tmp_path):
        """Barcodes, indices, cell and probe metadata, and run attributes are written."""
        out = _output_dir(tmp_path)
        path = _collect(out)
        with h5py.File(path, "r") as f:
            m = f["matrix"]
            assert [b.decode() for b in m["barcode"][:]] == [
                f"{c}-1" for c in support.CELLS[:3]
            ]
            assert m["cell_index"][:].tolist() == [0, 1, 2]
            assert [tuple(x.decode() for x in p) for p in m["probe"][:]] == [
                (A, "TAG"),
                (A, "TTG"),
                (B, "GCAT"),
            ]
            assert m["probe_index"][:].tolist() == [0, 0, 1]
            assert [c.decode() for c in f["cell_metadata"]["columns"][:]] == [
                "barcode",
                "plex_id",
                "plex_seq",
            ]
            assert f["cell_metadata"]["plex_id"][:].tolist() == [1, 1, 1]
            assert f["cell_metadata"]["plex_seq"][0] == b"ACTTTAGG"
            pm = f["probe_metadata"]
            assert [n.decode() for n in pm["name"][:]] == [A, B]
            assert [n.decode() for n in pm["original_sequence"][:]] == ["TAG", "GCCT"]
            assert pm["index"][:].tolist() == [0, 1]
            assert dict(f.attrs) | {"created_date": None} == {
                "plex": "1",
                "project": "run",
                "created_date": None,
                "n_cells": 3,
                "n_probes": 2,
                "n_probe_gapfill_combinations": 3,
                "max_pcr_duplicates": 4,
            }

    def test_pcr_threshold_layers_keep_umis_with_at_least_k_reads(self, tmp_path):
        """Layer `pcr<k>` counts only the UMIs supported by at least k reads."""
        path = _collect(_output_dir(tmp_path))
        expected = {
            1: [[0, 2, 1], [1, 0, 0], [0, 0, 0]],
            2: [[0, 1, 1], [0, 0, 0], [0, 0, 0]],
            3: [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            4: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }
        with h5py.File(path, "r") as f:
            assert sorted(f["pcr_thresholded_counts"].keys()) == [
                "pcr1",
                "pcr2",
                "pcr3",
                "pcr4",
            ]
            assert f["pcr_thresholded_counts"].attrs["max_pcr_duplicates"] == 4
        for k, matrix in expected.items():
            np.testing.assert_array_equal(
                _layer(path, f"pcr{k}", "pcr_thresholded_counts"), matrix
            )

    def test_threshold_count_is_capped_and_can_be_disabled(self, tmp_path):
        """`max_pcr_thresholds` caps the number of PCR layers, and 1 disables them."""
        capped = _collect(_output_dir(tmp_path, name="a"), max_pcr_thresholds=2)
        with h5py.File(capped, "r") as f:
            assert sorted(f["pcr_thresholded_counts"].keys()) == ["pcr1", "pcr2"]
        disabled = _collect(_output_dir(tmp_path, name="b"), max_pcr_thresholds=1)
        with h5py.File(disabled, "r") as f:
            assert "pcr_thresholded_counts" not in f
            assert f.attrs["max_pcr_duplicates"] == 1

    def test_only_reads_from_the_requested_plex_and_listed_cells_are_counted(
        self, tmp_path
    ):
        """Only reads from the requested plex and its listed cells are counted."""
        out = _output_dir(tmp_path, plexes=(1, 2, 2))
        barcodes = read_barcodes(out)
        step4.collect_counts(
            out / "probe_reads.tsv.gz",
            out,
            read_manifest(out),
            barcodes[barcodes.plex_id == 2],
            False,
            2,
            True,
            False,
            10,
        )
        adata = read_h5_file(out / "counts.2.h5")
        assert adata.obs_names.tolist() == [
            f"{support.CELLS[1]}-1",
            f"{support.CELLS[2]}-1",
        ]
        assert adata.obs["cell_index"].tolist() == [1, 2]
        assert adata.var_names.tolist() == [f"{A}|TAG"]
        assert adata.X.toarray().tolist() == [[1], [0]]
        assert adata.uns["plex"] == "2"

    def test_existing_output_requires_overwrite(self, tmp_path):
        """An existing counts file is kept unless `overwrite` is set."""
        out = _output_dir(tmp_path)
        path = _collect(out)
        with pytest.raises(AssertionError, match="already exists"):
            _collect(out)
        os.utime(path, (0, 0))
        _collect(out, overwrite=True)
        assert path.stat().st_mtime > 0

    def test_flatten_compiles_flat_file_for_the_plex(self, tmp_path, monkeypatch):
        """`flatten` writes `flat_counts.<plex>.tsv.gz` from the plex's reads and barcodes."""
        calls = []
        monkeypatch.setattr(step4, "compile_flatfile", lambda *a: calls.append(a))
        out = _output_dir(tmp_path)
        _collect(out, flatten=True)
        manifest, reads, barcodes, plex, target = calls[0]
        assert list(manifest["name"]) == [A, B]
        assert reads == str(out / "probe_reads.tsv.gz")
        assert barcodes == [f"{c}-1" for c in support.CELLS[:3]]
        assert (plex, target) == ("1", str(out / "flat_counts.1.tsv.gz"))

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=ValueError,
        reason="max() over an empty mapping when a plex has no valid reads",
    )
    def test_plex_without_reads_produces_an_empty_count_file(self, tmp_path):
        """A plex without valid reads still produces a counts file with no features."""
        path = _collect(_output_dir(tmp_path, reads=[[0, 0, 2, "U1", "TTG", 1, 1.0]]))
        assert read_h5_file(path).shape == (3, 0)


class TestRun:
    def _run(self, out, **kwargs):
        args = (
            dict(
                cores=1,
                overwrite=False,
                was_multiplexed=False,
                flatten=False,
                max_pcr_thresholds=10,
            )
            | kwargs
        )
        with pytest.raises(SystemExit) as exc:
            step4.run(str(out), **args)
        assert exc.value.code == 0

    def test_single_plex_run_writes_one_count_file(self, tmp_path, capsys):
        """A single-plex run writes `counts.1.h5`."""
        out = _output_dir(tmp_path)
        self._run(out)
        assert read_h5_file(out / "counts.1.h5").shape == (3, 3)
        assert "Counts data saved as counts.1.h5." in capsys.readouterr().out

    def test_single_plex_with_non_default_barcode_is_reported(self, tmp_path, capsys):
        """A single plex on another barcode writes `counts.<plex>.h5` and names the barcode."""
        reads = [[0, 0, 3, "U1", "TTG", 1, 1.0]]
        out = _output_dir(tmp_path, plexes=(3,), reads=reads)
        self._run(out, was_multiplexed=True)
        assert read_h5_file(out / "counts.3.h5").X.toarray().tolist() == [[1]]
        assert "single-plex run using BC3" in capsys.readouterr().out

    def test_uncompressed_reads_and_cpu_count_for_non_positive_cores(
        self, tmp_path, monkeypatch
    ):
        """An uncompressed reads file is accepted, and zero cores means all CPUs."""
        monkeypatch.setattr(os, "cpu_count", lambda: None)
        out = _output_dir(tmp_path)
        (out / "probe_reads.tsv").write_text(
            "\n".join(support.read_lines(out / "probe_reads.tsv.gz")) + "\n"
        )
        (out / "probe_reads.tsv.gz").unlink()
        self._run(out, cores=0)
        assert (out / "counts.1.h5").exists()

    @pytest.mark.multiprocess
    def test_multiplexed_run_writes_one_file_per_plex(self, tmp_path, capsys):
        """A multiplexed run on two cores writes one counts file per plex."""
        out = _output_dir(tmp_path, plexes=(1, 2, 2))
        self._run(out, cores=2)
        assert read_h5_file(out / "counts.1.h5").n_obs == 1
        assert read_h5_file(out / "counts.2.h5").n_obs == 2
        assert "counts.[1,2].h5" in capsys.readouterr().out

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="with cores=1 the lazy itertools.starmap result is discarded, so nothing is written"
    )
    def test_multiplexed_run_on_one_core_writes_one_file_per_plex(self, tmp_path):
        """A multiplexed run on one core also writes one counts file per plex."""
        out = _output_dir(tmp_path, plexes=(1, 2, 2))
        self._run(out, cores=1)
        assert (out / "counts.1.h5").exists() and (out / "counts.2.h5").exists()

    def test_missing_directory_or_reads_raise(self, tmp_path):
        """A missing output directory or reads file raises AssertionError."""
        with pytest.raises(AssertionError, match="Output directory does not exist"):
            step4.run(str(tmp_path / "nope"), 1, False, False, False, 10)
        with pytest.raises(AssertionError, match="Input file not found"):
            step4.run(str(tmp_path), 1, False, False, False, 10)


class TestMain:
    def test_arguments_are_forwarded(self, monkeypatch):
        """Options map to `run(output, cores, overwrite, multiplex, flatten, max_pcr_thresholds)`."""
        calls = []
        monkeypatch.setattr(step4, "run", lambda *a: calls.append(a))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "giftwrap-collect",
                "-o",
                "out",
                "-c",
                "2",
                "-m",
                "-f",
                "--flatten",
                "--max_pcr_thresholds",
                "5",
            ],
        )
        step4.main()
        monkeypatch.setattr(sys, "argv", ["giftwrap-collect", "--output", "out"])
        step4.main()
        assert calls == [
            ("out", 2, True, True, True, 5),
            ("out", 1, False, False, False, 10),
        ]

    def test_version(self, monkeypatch, capsys):
        """`--version` prints the package version and exits 0."""
        monkeypatch.setattr(sys, "argv", ["giftwrap-collect", "--version"])
        with pytest.raises(SystemExit) as exc:
            step4.main()
        assert exc.value.code == 0 and giftwrap.__version__ in capsys.readouterr().out
