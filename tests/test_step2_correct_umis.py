import gzip
import os
import sys

import pytest

import giftwrap
import giftwrap.step2_correct_umis as step2
import support
from giftwrap.step2_correct_umis import barcode_lines_generator, process_lines

U0, U1, U2 = support.UMIS[:3]
HEADER = [
    "cell_idx",
    "probe_idx",
    "probe_barcode",
    "gapfill",
    "gapfill_quality",
    "umi",
    "umi_quality",
]


def line(cell, probe, bc, umi, gap="TTG"):
    return f"{cell}\t{probe}\t{bc}\t{gap}\t{'I' * len(gap)}\t{umi}\t{'I' * len(umi)}\n"


class TestProcessLines:
    def test_close_umis_are_collapsed_onto_the_most_supported_umi(self):
        """A UMI one mismatch from a better-supported UMI is rewritten to it and counted."""
        lines = (
            [line(0, 0, 1, U0)] * 3
            + [line(0, 0, 1, support.mutate(U0, 4))]
            + [line(0, 1, 1, U1)] * 2
        )
        out, corrected, dropped = process_lines(
            lines, threshold=1, allow_chimeras=False
        )
        assert (corrected, dropped) == (1, 0)
        assert (
            out == [f"0\t1\t1\tTTG\tIII\t{U1}"] * 2 + [f"0\t0\t1\tTTG\tIII\t{U0}"] * 4
        )

    def test_distance_threshold_scales_with_umi_length(self):
        """For 12-bp UMIs, threshold 1 allows two mismatches while threshold 0 allows one."""
        two_off = support.mutate(support.mutate(U0, 1), 7)
        lines = [line(0, 0, 1, U0)] * 2 + [line(0, 0, 1, two_off)]
        assert process_lines(lines, threshold=1, allow_chimeras=True)[1] == 1
        assert process_lines(lines, threshold=0, allow_chimeras=True)[1] == 0

    def test_umis_are_only_compared_within_the_same_probe_barcode(self):
        """Similar UMIs under different probe barcodes are not corrected onto each other."""
        lines = [line(0, 0, 1, U0)] * 2 + [line(0, 0, 2, support.mutate(U0, 4))]
        out, corrected, _ = process_lines(lines, threshold=1, allow_chimeras=False)
        assert corrected == 0
        assert out[-1].endswith(support.mutate(U0, 4))

    def test_chimeric_umis_keep_only_the_majority_probe(self):
        """A UMI seen on several probes keeps its most frequent probe; other reads count as dropped."""
        lines = [line(0, 0, 1, U2)] * 3 + [line(0, 1, 1, U2, gap="GCAT")]
        out, corrected, dropped = process_lines(
            lines, threshold=1, allow_chimeras=False
        )
        assert (corrected, dropped) == (0, 1)
        assert out == [f"0\t0\t1\tTTG\tIII\t{U2}"] * 3

    def test_chimeras_are_kept_when_allowed(self):
        """With `allow_chimeras`, every probe of a shared UMI is kept."""
        lines = [line(0, 0, 1, U2)] * 3 + [line(0, 1, 1, U2, gap="GCAT")]
        out, _, dropped = process_lines(lines, threshold=1, allow_chimeras=True)
        assert dropped == 0
        assert len(out) == 4

    def test_blank_lines_are_ignored_and_output_sorted_by_barcode_cell_umi(self):
        """Blank lines are skipped and output is sorted by probe barcode, cell and UMI."""
        lines = ["\n", line(1, 0, 2, U1), line(0, 0, 2, U0), line(0, 0, 1, U1)]
        out, _, _ = process_lines(lines, threshold=0, allow_chimeras=True)
        assert [tuple(row.split("\t")[i] for i in (2, 0, 5)) for row in out] == [
            ("1", "0", U1),
            ("2", "0", U0),
            ("2", "1", U1),
        ]

    def test_empty_input(self):
        """No lines give no output, no corrections and no drops."""
        assert process_lines([], threshold=1, allow_chimeras=False) == ([], 0, 0)


class TestBarcodeLinesGenerator:
    def test_groups_consecutive_lines_by_cell(self):
        """Consecutive lines of one cell form a group, even if the cell recurs later."""
        lines = [
            line(0, 0, 1, U0),
            line(0, 1, 1, U1),
            line(1, 0, 1, U0),
            line(0, 0, 1, U2),
        ]
        groups = list(barcode_lines_generator(iter(lines)))
        assert groups == [(lines[:2],), (lines[2:3],), (lines[3:],)]

    def test_empty_input_yields_nothing(self):
        """An empty file yields no groups."""
        assert list(barcode_lines_generator(iter([]))) == []


def _write_reads(directory, rows, name="probe_reads.tsv.gz"):
    return support.write_lines(
        directory / name, ["\t".join(HEADER)] + [r.rstrip("\n") for r in rows]
    )


class TestRun:
    ROWS = [
        line(0, 0, 1, U0),
        line(0, 0, 1, U0),
        line(0, 0, 1, support.mutate(U0, 3)),
        line(1, 1, 1, U1),
    ]

    def _run(self, tmp_path, **kwargs):
        args = (
            dict(threshold=1, cores=1, n_cells_per_batch=1, allow_chimeras=False)
            | kwargs
        )
        with pytest.raises(SystemExit) as exc:
            step2.run(str(tmp_path), **args)
        assert exc.value.code == 0

    def test_rewrites_reads_with_corrected_umis_and_keeps_backup(
        self, tmp_path, capsys
    ):
        """The reads file is rewritten without UMI qualities, with a `.bak.umi` backup and mode 0o766."""
        path = _write_reads(tmp_path, self.ROWS)
        original = path.read_bytes()
        self._run(tmp_path)
        assert support.read_lines(path) == [
            "cell_idx\tprobe_idx\tprobe_barcode\tgapfill\tgapfill_quality\tumi",
            *[f"0\t0\t1\tTTG\tIII\t{U0}"] * 3,
            f"1\t1\t1\tTTG\tIII\t{U1}",
        ]
        assert (tmp_path / "probe_reads.tsv.bak.umi").read_bytes() == original
        assert oct(os.stat(path).st_mode & 0o777) == oct(0o766)
        assert "1 UMIs corrected." in capsys.readouterr().out

    def test_previous_backup_is_replaced(self, tmp_path):
        """An existing `.bak.umi` backup is replaced by the current input."""
        _write_reads(tmp_path, self.ROWS)
        (tmp_path / "probe_reads.tsv.bak.umi").write_text("stale")
        self._run(tmp_path, n_cells_per_batch=10)
        with gzip.open(tmp_path / "probe_reads.tsv.bak.umi", "rt") as f:
            assert f.readline().startswith("cell_idx")

    def test_non_positive_cores_use_cpu_count(self, tmp_path, monkeypatch):
        """Zero cores means all CPUs, or one core when the CPU count is unknown."""
        monkeypatch.setattr(os, "cpu_count", lambda: None)
        _write_reads(tmp_path, self.ROWS)
        self._run(tmp_path, cores=0)
        assert len(support.read_lines(tmp_path / "probe_reads.tsv.gz")) == 5

    @pytest.mark.multiprocess
    def test_multiprocess_matches_single_process_output(self, tmp_path):
        """Two worker processes write the same file as one."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        rows = self.ROWS + [line(2, 0, 1, U2), line(3, 1, 1, U0)]
        _write_reads(tmp_path / "a", rows)
        _write_reads(tmp_path / "b", rows)
        self._run(tmp_path / "a", cores=1)
        self._run(tmp_path / "b", cores=2)
        assert support.read_lines(
            tmp_path / "a" / "probe_reads.tsv.gz"
        ) == support.read_lines(tmp_path / "b" / "probe_reads.tsv.gz")

    def test_missing_output_or_input_raise(self, tmp_path):
        """A missing output directory or reads file raises AssertionError."""
        with pytest.raises(AssertionError, match="Output directory"):
            step2.run(str(tmp_path / "nope"), 1, 1, 1, False)
        with pytest.raises(AssertionError, match="Input file not found"):
            step2.run(str(tmp_path), 1, 1, 1, False)

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=UnicodeDecodeError,
        reason="an uncompressed probe_reads.tsv is replaced by gzip-compressed bytes",
    )
    def test_uncompressed_input_stays_readable_as_text(self, tmp_path):
        """An uncompressed `probe_reads.tsv` stays plain text after correction."""
        path = _write_reads(tmp_path, self.ROWS, name="probe_reads.tsv")
        self._run(tmp_path)
        assert support.read_lines(path)[0].startswith("cell_idx")


class TestMain:
    def test_arguments_are_forwarded_to_run(self, monkeypatch):
        """Short options map to `run(output, threshold, cores, n_cells_per_batch, allow_chimeras)`."""
        calls = []
        monkeypatch.setattr(step2, "run", lambda *a: calls.append(a))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "giftwrap-correct-umis",
                "-o",
                "out",
                "-t",
                "2",
                "-c",
                "3",
                "-n",
                "5",
                "-ac",
            ],
        )
        step2.main()
        assert calls == [("out", 2, 3, 5, True)]

    def test_defaults(self, monkeypatch):
        """Defaults are threshold 1, one core, 1,000 cells per batch and no chimeras."""
        calls = []
        monkeypatch.setattr(step2, "run", lambda *a: calls.append(a))
        monkeypatch.setattr(sys, "argv", ["giftwrap-correct-umis", "--output", "out"])
        step2.main()
        assert calls == [("out", 1, 1, 1000, False)]

    def test_version_and_missing_output(self, monkeypatch, capsys):
        """`--version` exits 0 and a missing `--output` is a usage error (exit 2)."""
        monkeypatch.setattr(sys, "argv", ["giftwrap-correct-umis", "--version"])
        with pytest.raises(SystemExit) as exc:
            step2.main()
        assert exc.value.code == 0 and giftwrap.__version__ in capsys.readouterr().out
        monkeypatch.setattr(sys, "argv", ["giftwrap-correct-umis"])
        with pytest.raises(SystemExit) as exc:
            step2.main()
        assert exc.value.code == 2
