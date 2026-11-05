import gzip
import io
import itertools
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import giftwrap
import giftwrap.pipeline_utils as pu
import support
from giftwrap.pipeline_utils import (
    PIPELINE_INVOCATION_FILE,
    GzipNamedTemporaryFile,
    ReadProcessState,
    _skip_manifest_header,
    batched,
    compile_flatfile,
    format_command_line,
    lru_cached_method,
    maybe_gzip,
    read_barcodes,
    read_fastqs,
    read_manifest,
    read_manifest_config,
    read_pipeline_invocations,
    read_probes_input,
    record_pipeline_invocation,
    sort_tsv_file,
    write_manifest,
    write_sparse_matrix,
)


def test_read_process_state_members_and_values():
    """Read states keep their names and numeric values (TOTAL_READS is 10, EXACT is 9)."""
    assert ReadProcessState.TOTAL_READS.value == 10
    assert ReadProcessState.EXACT.value == 9
    assert {s.name for s in ReadProcessState} >= {
        "FILTERED_NO_CONSTANT",
        "FILTERED_NO_RHS",
        "FILTERED_NO_LHS",
        "FILTERED_NO_PROBE_BARCODE",
        "FILTERED_NO_CELL_BARCODE",
        "CORRECTED_RHS",
        "CORRECTED_LHS",
        "CORRECTED_BARCODE",
    }


class TestBatched:
    def test_chunks_an_iterator_with_short_final_batch(self):
        """An iterator is split into lists of n items; the last list may be shorter."""
        assert list(batched(iter(range(5)), 2)) == [[0, 1], [2, 3], [4]]

    def test_exact_multiple_and_empty_input(self):
        """Exact multiples give only full batches, and an empty iterator gives none."""
        assert list(batched(iter("abcd"), 2)) == [["a", "b"], ["c", "d"]]
        assert list(batched(iter([]), 3)) == []

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="batched restarts islice on a re-iterable (e.g. list) and never terminates"
    )
    def test_accepts_a_reiterable_sequence(self):
        """A list, not only an iterator, is batched and the generator terminates."""
        assert list(itertools.islice(batched([0, 1, 2], 2), 3)) == [[0, 1], [2]]


class TestLruCachedMethod:
    class Counter:
        def __init__(self):
            self.calls = 0

        @lru_cached_method(maxsize=1)
        def square(self, x):
            self.calls += 1
            return x * x

    def test_caches_per_instance_and_binds_cache_to_the_instance(self):
        """Each instance gets its own cache, stored on the instance after first access."""
        a, b = self.Counter(), self.Counter()
        assert a.square(3) == 9
        assert a.square(3) == 9
        assert a.calls == 1
        assert "square" in a.__dict__
        assert b.square(3) == 9
        assert b.calls == 1

    def test_respects_maxsize(self):
        """Entries are evicted once more than `maxsize` arguments have been seen."""
        a = self.Counter()
        a.square(1)
        a.square(2)
        a.square(1)
        assert a.calls == 3

    def test_class_access_returns_descriptor_with_wrapped_metadata(self):
        """Class-level access returns the descriptor, which carries the function's name."""
        descriptor = self.Counter.__dict__["square"]
        assert self.Counter.square is descriptor
        assert descriptor.__name__ == "square"


class TestManifest:
    def test_round_trip_with_json_config_header(self, tmp_path):
        """Config is written as `# key=<json>` lines and read back with the table intact."""
        df = support.manifest_df()
        config = {
            "technology": "Flex",
            "multiplex": 2,
            "barcodes": ["1", "2"],
            "tech_def": None,
        }
        write_manifest(tmp_path, df, config)
        lines = (tmp_path / "manifest.tsv").read_text().splitlines()
        assert lines[0] == '# technology="Flex"'
        assert lines[3] == "# tech_def=null"
        assert read_manifest_config(tmp_path) == config
        pd.testing.assert_frame_equal(read_manifest(tmp_path), df)

    def test_without_config_there_is_no_header(self, tmp_path):
        """Without a config, the file starts with the table header and the config reads as {}."""
        write_manifest(str(tmp_path), support.manifest_df(), None)
        assert (tmp_path / "manifest.tsv").read_text().startswith("name\t")
        assert read_manifest_config(str(tmp_path)) == {}
        assert read_manifest(str(tmp_path)).shape == (2, 7)

    def test_config_values_that_are_not_json_are_kept_as_strings_and_lines_without_equals_skipped(
        self, tmp_path
    ):
        """Non-JSON values stay strings, keys are stripped, and lines without '=' are ignored."""
        (tmp_path / "manifest.tsv").write_text(
            "# tech=Flex\r\n# orphan\n# spaced key = 3\nname\tindex\nx\t0\n"
        )
        assert read_manifest_config(tmp_path) == {"tech": "Flex", "spaced key": 3}
        assert read_manifest(tmp_path).to_dict("records") == [{"name": "x", "index": 0}]

    def test_skip_header_leaves_handle_on_table_header(self):
        """Header lines are returned without their prefix and the handle stops at the table."""
        handle = io.StringIO("# a=1\n# b=2\ncol\n1\n")
        assert _skip_manifest_header(handle) == ["a=1", "b=2"]
        assert handle.readline() == "col\n"

    def test_skip_header_without_header_or_content(self):
        """Without header lines, nothing is consumed from the handle."""
        assert _skip_manifest_header(io.StringIO("")) == []
        handle = io.StringIO("col\n")
        assert _skip_manifest_header(handle) == []
        assert handle.read() == "col\n"

    def test_unserializable_config_raises(self, tmp_path):
        """Config values that are not JSON-serializable raise TypeError."""
        with pytest.raises(TypeError):
            write_manifest(tmp_path, support.manifest_df(), {"bad": object()})


class TestCommandLine:
    def test_empty_argv(self):
        """An empty argv renders as an empty string."""
        assert format_command_line([]) == ""

    def test_program_reduced_to_basename_and_arguments_shell_quoted(self, monkeypatch):
        """The program path is reduced to its name and arguments are POSIX-shell quoted."""
        monkeypatch.setattr(sys, "platform", "linux")
        argv = ["/opt/venv/bin/giftwrap", "-o", "out dir", 3]
        assert format_command_line(argv) == "giftwrap -o 'out dir' 3"

    @pytest.mark.parametrize(
        "prog", ["giftwrap.exe", "giftwrap-script.py", "giftwrap.py", "GIFTWRAP.EXE"]
    )
    def test_launcher_suffixes_are_stripped(self, prog, monkeypatch):
        """'.exe', '-script.py' and '.py' suffixes are removed, case-insensitively."""
        monkeypatch.setattr(sys, "platform", "linux")
        assert format_command_line([prog]).lower() == "giftwrap"

    def test_windows_uses_list2cmdline_quoting(self, monkeypatch):
        """On Windows, arguments are quoted with `subprocess.list2cmdline`."""
        monkeypatch.setattr(sys, "platform", "win32")
        assert (
            format_command_line(["giftwrap", "-o", "out dir"])
            == 'giftwrap -o "out dir"'
        )


class TestPipelineInvocations:
    def test_record_creates_and_appends_history(self, tmp_path, monkeypatch):
        """Each call appends a full record to `pipeline_invocation.json`, oldest first."""
        monkeypatch.chdir(tmp_path)
        first = record_pipeline_invocation(tmp_path, ["giftwrap", "-o", "out"])
        second = record_pipeline_invocation(str(tmp_path), ["giftwrap", "-f", 1])
        assert first["command"] == "giftwrap -o out"
        assert first["argv"] == ["giftwrap", "-o", "out"]
        assert first["cwd"] == str(tmp_path)
        assert first["platform"] == sys.platform
        assert first["giftwrap_version"] == giftwrap.__version__
        assert set(first) == {
            "timestamp",
            "giftwrap_version",
            "command",
            "argv",
            "cwd",
            "platform",
        }
        assert second["argv"] == ["giftwrap", "-f", "1"]
        stored = json.loads((tmp_path / PIPELINE_INVOCATION_FILE).read_text())
        assert stored == [first, second]
        assert read_pipeline_invocations(tmp_path) == [first, second]

    def test_record_defaults_to_sys_argv_and_tolerates_missing_package_module(
        self, tmp_path, monkeypatch
    ):
        """Without argv, `sys.argv` is recorded; without the giftwrap module, the version is None."""
        monkeypatch.setattr(sys, "argv", ["/bin/giftwrap", "--cores", "2"])
        monkeypatch.delitem(sys.modules, "giftwrap")
        record = record_pipeline_invocation(tmp_path)
        assert record["command"] == "giftwrap --cores 2"
        assert record["giftwrap_version"] is None

    def test_read_missing_file_is_empty(self, tmp_path):
        """Without an invocation file, the history is empty."""
        assert read_pipeline_invocations(tmp_path) == []

    def test_read_corrupt_or_unreadable_file_is_empty(self, tmp_path):
        """Invalid JSON or an unreadable path gives an empty history instead of an error."""
        (tmp_path / PIPELINE_INVOCATION_FILE).write_text("{not json")
        assert read_pipeline_invocations(tmp_path) == []
        other = tmp_path / "other"
        (other / PIPELINE_INVOCATION_FILE).mkdir(parents=True)
        assert read_pipeline_invocations(other) == []

    def test_read_tolerates_single_record_and_drops_invalid_entries(self, tmp_path):
        """A lone record is wrapped in a list; entries without a command are dropped."""
        (tmp_path / PIPELINE_INVOCATION_FILE).write_text(
            json.dumps({"command": "giftwrap"})
        )
        assert read_pipeline_invocations(tmp_path) == [{"command": "giftwrap"}]
        (tmp_path / PIPELINE_INVOCATION_FILE).write_text(
            json.dumps([1, {"command": ""}, {"x": 1}, {"command": "ok"}])
        )
        assert read_pipeline_invocations(tmp_path) == [{"command": "ok"}]


class TestReadBarcodes:
    def test_plain_file_is_preferred_over_gzip(self, tmp_path):
        """`barcodes.tsv` takes precedence over `barcodes.tsv.gz`."""
        support.write_tsv(
            tmp_path / "barcodes.tsv", ["barcode", "plex_id"], [["A-1", 1]]
        )
        support.write_tsv(
            tmp_path / "barcodes.tsv.gz",
            ["barcode", "plex_id"],
            [["B-1", 1], ["C-1", 1]],
        )
        assert read_barcodes(str(tmp_path)).barcode.tolist() == ["A-1"]

    def test_gzip_file(self, tmp_path):
        """A gzipped barcodes file is read with its columns."""
        support.write_tsv(
            tmp_path / "barcodes.tsv.gz",
            ["barcode", "plex_id"],
            [["B-1", 1], ["C-1", 2]],
        )
        df = read_barcodes(tmp_path)
        assert df.to_dict("list") == {"barcode": ["B-1", "C-1"], "plex_id": [1, 2]}

    def test_missing_file_raises(self, tmp_path):
        """Without a barcodes file, FileNotFoundError is raised."""
        with pytest.raises(FileNotFoundError):
            read_barcodes(tmp_path)


class TestGzipHandles:
    def test_named_temporary_gzip_file_writes_text_and_persists(self):
        """Text written through the wrapper is gzip-compressed and kept after closing."""
        with GzipNamedTemporaryFile() as f:
            f.write("a\tb\n")
            f.write("c\n")
            name = f.name
        try:
            with gzip.open(name, "rt") as handle:
                assert handle.read() == "a\tb\nc\n"
        finally:
            os.remove(name)

    def test_maybe_gzip_without_path_returns_temporary_writer_or_raises_for_read(self):
        """Without a path, write mode gives a temporary gzip writer and read mode raises."""
        writer = maybe_gzip(None, "w")
        assert isinstance(writer, GzipNamedTemporaryFile)
        with writer:
            pass
        os.remove(writer.name)
        with pytest.raises(ValueError):
            maybe_gzip(None, "r")

    @pytest.mark.parametrize("name", ["x.tsv", "x.tsv.gz"])
    def test_maybe_gzip_round_trips_plain_and_gzip(self, tmp_path, name):
        """A '.gz' suffix selects gzip compression for both writing and reading."""
        path = tmp_path / name
        with maybe_gzip(path, "w") as f:
            f.write("hello\n")
        with maybe_gzip(str(path), "r") as f:
            assert f.read() == "hello\n"
        with open(path, "rb") as raw:
            assert (raw.read(2) == b"\x1f\x8b") == name.endswith(".gz")


class TestReadProbesInput:
    def test_tsv_with_standard_columns(self, tmp_path):
        """A TSV panel yields the standard manifest columns in order with a fresh index."""
        df = read_probes_input(str(support.write_probe_file(tmp_path / "probes.tsv")))
        assert df.columns.tolist() == [
            "name",
            "lhs_probe",
            "rhs_probe",
            "gap_probe_sequence",
            "original_gap_probe_sequence",
            "gene",
        ]
        assert df.name.tolist() == ["GENEA c.1A>T", "GENEB c.2C>G"]
        assert df.index.tolist() == [0, 1]

    def test_csv_with_alias_columns_and_mixed_case_headers(self, tmp_path):
        """Headers are lowercased and expected/reference/symbol aliases are renamed."""
        raw = pd.DataFrame(
            {
                "NAME": ["p1"],
                "LHS_Probe": [support.LHS_A],
                "RHS_PROBE": [support.RHS_A],
                "Expected_Gapfill": ["TTG"],
                "Reference_Gapfill": ["TAG"],
                "Symbol": ["GENEA"],
            }
        )
        raw.to_csv(tmp_path / "probes.csv", index=False)
        df = read_probes_input(tmp_path / "probes.csv")
        assert df.to_dict("records") == [
            {
                "name": "p1",
                "lhs_probe": support.LHS_A,
                "rhs_probe": support.RHS_A,
                "gap_probe_sequence": "TTG",
                "original_gap_probe_sequence": "TAG",
                "gene": "GENEA",
            }
        ]

    @pytest.mark.parametrize("alias", ["gene_name", "gene_symbol"])
    def test_other_gene_column_aliases(self, tmp_path, alias):
        """`gene_name` and `gene_symbol` columns are renamed to `gene`."""
        raw = pd.DataFrame(
            {"name": ["p1"], "lhs_probe": ["AC"], "rhs_probe": ["GT"], alias: ["G1"]}
        )
        raw.to_csv(tmp_path / "p.csv", index=False)
        assert read_probes_input(tmp_path / "p.csv")["gene"].tolist() == ["G1"]

    def test_missing_gap_columns_default_to_na_and_no_gene_column(self, tmp_path):
        """Missing gap columns default to "NA", and no gene column is added."""
        raw = pd.DataFrame({"name": ["p1"], "lhs_probe": ["AC"], "rhs_probe": ["GT"]})
        raw.to_csv(tmp_path / "p.tsv", sep="\t", index=False)
        df = read_probes_input(tmp_path / "p.tsv")
        assert df.columns.tolist() == [
            "name",
            "lhs_probe",
            "rhs_probe",
            "gap_probe_sequence",
            "original_gap_probe_sequence",
        ]
        assert df.iloc[0]["gap_probe_sequence"] == "NA"
        assert df.iloc[0]["original_gap_probe_sequence"] == "NA"

    def test_exact_duplicates_dropped_and_duplicate_names_suffixed(
        self, tmp_path, capsys
    ):
        """Exact duplicates are dropped; remaining name clashes get _0, _1 suffixes and a warning."""
        raw = pd.DataFrame(
            {
                "name": ["p", "p", "p", "q"],
                "lhs_probe": ["AC", "AC", "AG", "TT"],
                "rhs_probe": ["GT", "GT", "GT", "CC"],
            }
        )
        raw.to_csv(tmp_path / "p.csv", index=False)
        df = read_probes_input(tmp_path / "p.csv")
        assert df.name.tolist() == ["p_0", "p_1", "q"]
        assert df.lhs_probe.tolist() == ["AC", "AG", "TT"]
        assert "Duplicated probe names" in capsys.readouterr().out

    def test_excel_input(self, tmp_path):
        """Excel panels are read like CSV and TSV ones."""
        df = read_probes_input(support.write_probe_file(tmp_path / "probes.xlsx"))
        assert df.name.tolist() == ["GENEA c.1A>T", "GENEB c.2C>G"]

    def test_missing_required_columns_raise(self, tmp_path):
        """A panel without both probe arms raises KeyError."""
        pd.DataFrame({"name": ["p"], "lhs_probe": ["A"]}).to_csv(
            tmp_path / "p.csv", index=False
        )
        with pytest.raises(KeyError):
            read_probes_input(tmp_path / "p.csv")


class TestReadFastqs:
    def test_chains_paired_records_across_plain_and_gzip_files(self, tmp_path):
        """Records from several file pairs, plain or gzipped, are chained in order."""
        r1a = support.write_fastq(tmp_path / "a_R1.fastq.gz", [("r0", "AC", "II")])
        r2a = support.write_fastq(tmp_path / "a_R2.fastq.gz", [("r0", "GT", "II")])
        r1b = support.write_fastq(
            tmp_path / "b_R1.fastq", [("r1", "AAA", "III"), ("r2", "C", "I")]
        )
        r2b = support.write_fastq(
            tmp_path / "b_R2.fastq", [("r1", "TTT", "III"), ("r2", "G", "I")]
        )
        with read_fastqs([str(r1a), str(r1b)], [str(r2a), str(r2b)]) as (it1, it2):
            pairs = list(zip(it1, it2))
        assert pairs == [
            (("r0", "AC", "II"), ("r0", "GT", "II")),
            (("r1", "AAA", "III"), ("r1", "TTT", "III")),
            (("r2", "C", "I"), ("r2", "G", "I")),
        ]

    def test_unbalanced_file_lists_raise(self, tmp_path):
        """Different numbers of R1 and R2 files raise ValueError."""
        r1 = support.write_fastq(tmp_path / "a_R1.fastq", [("r0", "AC", "II")])
        with pytest.raises(ValueError):
            with read_fastqs([str(r1), str(r1)], [str(r1)]):
                pass


def _unsorted_table(path):
    return support.write_tsv(
        path,
        ["cell_idx", "probe_idx", "probe_barcode"],
        [[2, 0, 1], [0, 1, 2], [1, 1, 1], [0, 0, 1], [3, 0, 2]],
    )


SORTED_BY_BC_CELL_PROBE = [
    "cell_idx\tprobe_idx\tprobe_barcode",
    "0\t0\t1",
    "1\t1\t1",
    "2\t0\t1",
    "0\t1\t2",
    "3\t0\t2",
]


class TestSortTsvFile:
    @pytest.mark.parametrize("name", ["reads.tsv", "reads.tsv.gz"])
    def test_single_core_sorts_with_pandas(self, tmp_path, name, monkeypatch):
        """On one core the table is sorted in place with pandas, without a shell command."""
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: pytest.fail("unix sort must not be used")
        )
        path = _unsorted_table(tmp_path / name)
        sort_tsv_file(path, [2, 0, 1], cores=1)
        assert support.read_lines(path) == SORTED_BY_BC_CELL_PROBE

    @pytest.mark.parametrize("name", ["reads.tsv", "reads.tsv.gz"])
    def test_multi_core_uses_unix_sort_and_keeps_header(self, tmp_path, name):
        """On several cores Unix `sort` orders the rows, keeps the header and leaves no temp file."""
        path = _unsorted_table(tmp_path / name)
        sort_tsv_file(path, [2, 0, 1], cores=2)
        assert support.read_lines(path) == SORTED_BY_BC_CELL_PROBE
        assert sorted(p.name for p in tmp_path.iterdir()) == [name]

    def test_use_pandas_flag_skips_unix_sort(self, tmp_path, monkeypatch):
        """`use_pandas=True` forces the pandas path even on several cores."""
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: pytest.fail("unix sort must not be used")
        )
        path = _unsorted_table(tmp_path / "reads.tsv.gz")
        sort_tsv_file(path, [2, 0, 1], cores=4, use_pandas=True)
        assert support.read_lines(path) == SORTED_BY_BC_CELL_PROBE

    def test_missing_sort_binary_falls_back_to_pandas(self, tmp_path, monkeypatch):
        """Without a `sort` binary on PATH, pandas sorts the file."""
        monkeypatch.setattr(pu.shutil, "which", lambda name: None)
        path = _unsorted_table(tmp_path / "reads.tsv.gz")
        sort_tsv_file(path, [2, 0, 1], cores=2)
        assert support.read_lines(path) == SORTED_BY_BC_CELL_PROBE

    def test_failed_unix_sort_restores_input_and_falls_back(
        self, tmp_path, monkeypatch
    ):
        """A failing `sort` restores the input, removes the temp file and falls back to pandas."""
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1)
        )
        path = _unsorted_table(tmp_path / "reads.tsv.gz")
        sort_tsv_file(path, [2, 0, 1], cores=2)
        assert support.read_lines(path) == SORTED_BY_BC_CELL_PROBE
        assert not path.with_suffix(".tmp").exists()

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="paths are interpolated unquoted into a shell command; a space creates a stray file"
    )
    def test_paths_with_spaces_do_not_leak_files_outside_target_directory(
        self, tmp_path
    ):
        """Paths containing spaces are sorted without creating files elsewhere."""
        target_dir = tmp_path / "my dir"
        target_dir.mkdir()
        path = _unsorted_table(target_dir / "reads.tsv.gz")
        sort_tsv_file(path, [2, 0, 1], cores=2)
        assert support.read_lines(path) == SORTED_BY_BC_CELL_PROBE
        assert sorted(p.name for p in tmp_path.iterdir()) == ["my dir"]


class TestWriteSparseMatrix:
    def test_writes_csr_components_shape_and_converts_other_formats(self, tmp_path):
        """Non-CSR input is stored as gzip-compressed CSR data/indices/indptr plus its shape."""
        mat = sp.coo_matrix(np.array([[0, 5], [7, 0], [0, 0]]))
        with h5py.File(tmp_path / "m.h5", "w") as f:
            write_sparse_matrix(f, "layer", mat)
        with h5py.File(tmp_path / "m.h5", "r") as f:
            grp = f["layer"]
            assert tuple(grp.attrs["shape"]) == (3, 2)
            assert grp["data"][:].tolist() == [5, 7]
            assert grp["indices"][:].tolist() == [1, 0]
            assert grp["indptr"][:].tolist() == [0, 1, 2, 2]
            assert grp["data"].compression == "gzip"


class TestCompileFlatfile:
    HEADER = [
        "cell_idx",
        "probe_idx",
        "probe_barcode",
        "umi",
        "gapfill",
        "pcr_duplicate_count",
        "percent_supporting",
    ]

    def test_writes_header_even_without_reads(self, tmp_path):
        """The flat file always starts with its header, even without reads."""
        reads = support.write_tsv(tmp_path / "probe_reads.tsv.gz", self.HEADER, [])
        compile_flatfile(
            support.manifest_df(),
            str(reads),
            ["AAAA-1"],
            "1",
            str(tmp_path / "flat.tsv.gz"),
        )
        assert support.read_lines(tmp_path / "flat.tsv.gz") == [
            "cell_barcode\tlhs_probe\trhs_probe\tcalled_probe\tgapfill\tpcr_duplicates\tpercent_supporting\tumi"
        ]

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="rows are skipped unless they have 6 columns (they have 7) and plex is compared int vs str"
    )
    def test_writes_one_row_per_umi_of_the_requested_plex(self, tmp_path):
        """Each UMI of the requested plex becomes one row keyed by its cell barcode."""
        reads = support.write_tsv(
            tmp_path / "probe_reads.tsv.gz",
            self.HEADER,
            [
                [0, 0, 1, "UMI1", "TTG", 3, 1.0],
                [1, 1, 1, "UMI2", "GCAT", 1, 0.5],
                [1, 1, 2, "UMI3", "GCAT", 1, 1.0],
            ],
        )
        compile_flatfile(
            support.manifest_df(),
            str(reads),
            ["AAAA-1", "CCCC-1"],
            "1",
            str(tmp_path / "flat.tsv.gz"),
        )
        rows = [
            line.split("\t")
            for line in support.read_lines(tmp_path / "flat.tsv.gz")[1:]
        ]
        assert [(r[0], r[4], r[5], r[7]) for r in rows] == [
            ("AAAA-1", "TTG", "3", "UMI1"),
            ("CCCC-1", "GCAT", "1", "UMI2"),
        ]
