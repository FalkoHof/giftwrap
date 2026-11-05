import gzip
import os
import sys

import pytest

import giftwrap
import giftwrap.step3_correct_gapfill as step3
import support
from giftwrap.step3_correct_gapfill import (
    barcode_umi_name_lines_generator,
    process_lines,
)

U0, U1 = support.UMIS[:2]


def row(gap, cell="0", probe="1", bc="1", umi=U0, q="I"):
    return [cell, probe, bc, gap, q * len(gap), umi]


class TestProcessLines:
    def test_single_read_is_passed_through(self):
        """A single read is written as-is with one duplicate and full support."""
        assert process_lines([row("TTG")]) == (f"0\t1\t1\t{U0}\tTTG\t1\t1.0\n", False)

    def test_identical_reads_are_collapsed_with_duplicate_count(self):
        """Identical gapfills collapse into one uncorrected row counting every read."""
        assert process_lines([row("TTG")] * 3) == (
            f"0\t1\t1\t{U0}\tTTG\t3\t1.0\n",
            False,
        )

    def test_length_disagreement_resolves_to_majority_length(self):
        """The most common gapfill length wins; support is its share of the reads."""
        out, corrected = process_lines([row("TTG"), row("TTG"), row("TT")])
        assert corrected is True
        assert out == f"0\t1\t1\t{U0}\tTTG\t3\t{2 / 3}\n"

    def test_majority_length_can_be_an_empty_gapfill(self):
        """An empty gapfill can be the majority length."""
        out, _ = process_lines([row(""), row(""), row("A")])
        assert out.split("\t")[4:6] == ["", "3"]

    def test_base_wise_consensus_uses_majority_base(self):
        """Disagreeing gapfills of equal length get a per-base majority consensus."""
        out, corrected = process_lines([row("TTG"), row("TTG"), row("TAG")])
        assert corrected is True
        assert out == f"0\t1\t1\t{U0}\tTTG\t3\t{2 / 3}\n"

    def test_consensus_after_length_filtering(self):
        """The consensus uses only reads of the majority length."""
        out, _ = process_lines([row("GCAT"), row("GCAT"), row("GCTT"), row("GC")])
        assert out == f"0\t1\t1\t{U0}\tGCAT\t4\t0.5\n"

    def test_ambiguous_bases_support_every_nucleotide(self):
        """An N counts toward all four bases at its position."""
        out, _ = process_lines([row("TNG"), row("TAG")])
        assert out.split("\t")[4] == "TAG"
        assert out.endswith("\t0.5\n")

    @pytest.mark.parametrize("base", "ACGT")
    def test_each_unanimous_base_is_called(self, base):
        """A position where all reads agree is called as that base, for each of A, C, G and T."""
        out, _ = process_lines([row(base + "A"), row(base + "C"), row(base + "C")])
        assert out.split("\t")[4] == base + "C"


class TestGroupingGenerator:
    def _line(self, cell, probe, bc, umi):
        return f"{cell}\t{probe}\t{bc}\tTTG\tIII\t{umi}\n"

    def test_groups_change_on_cell_probe_barcode_or_umi(self):
        """A new group starts whenever the cell, probe, probe barcode or UMI changes."""
        lines = [
            self._line(0, 0, 1, U0),
            self._line(0, 0, 1, U0),
            self._line(0, 0, 1, U1),
            self._line(0, 1, 1, U1),
            self._line(0, 1, 2, U1),
            self._line(1, 1, 2, U1),
            "\n",
        ]
        groups = [g[0] for g in barcode_umi_name_lines_generator(iter(lines))]
        assert [len(g) for g in groups] == [2, 1, 1, 1, 1]
        assert groups[0][0] == ["0", "0", "1", "TTG", "III", U0]

    def test_empty_input(self):
        """Blank lines alone yield no groups."""
        assert list(barcode_umi_name_lines_generator(iter(["", "\n"]))) == []


STEP2_HEADER = "cell_idx\tprobe_idx\tprobe_barcode\tgapfill\tgapfill_quality\tumi"
STEP3_HEADER = "cell_idx\tprobe_idx\tprobe_barcode\tumi\tgapfill\tpcr_duplicate_count\tpercent_supporting"


def _write_reads(directory, name="probe_reads.tsv.gz"):
    rows = [f"0\t0\t1\tTTG\tIII\t{U0}"] * 2 + [
        f"0\t0\t1\tTAG\tIII\t{U0}",
        f"1\t1\t1\tGCAT\tIIII\t{U1}",
    ]
    return support.write_lines(directory / name, [STEP2_HEADER] + rows)


class TestRun:
    def _run(self, path, cores=1, n=1):
        with pytest.raises(SystemExit) as exc:
            step3.run(str(path), cores, n)
        assert exc.value.code == 0

    def test_collapses_umis_into_consensus_rows_and_keeps_backup(
        self, tmp_path, capsys
    ):
        """Each UMI becomes one consensus row under the new header, with a backup and mode 0o766."""
        path = _write_reads(tmp_path)
        original = path.read_bytes()
        self._run(tmp_path)
        assert support.read_lines(path) == [
            STEP3_HEADER,
            f"0\t0\t1\t{U0}\tTTG\t3\t{2 / 3}",
            f"1\t1\t1\t{U1}\tGCAT\t1\t1.0",
        ]
        assert (tmp_path / "probe_reads.tsv.bak.gapfill").read_bytes() == original
        assert oct(os.stat(path).st_mode & 0o777) == oct(0o766)
        out = capsys.readouterr().out
        assert (
            "1 out of 2 gapfills corrected." in out
            and "2 unique counts identified." in out
        )

    def test_previous_backup_is_replaced_and_cpu_count_used_for_non_positive_cores(
        self, tmp_path, monkeypatch
    ):
        """An existing `.bak.gapfill` backup is replaced, and zero cores means all CPUs."""
        monkeypatch.setattr(os, "cpu_count", lambda: 1)
        _write_reads(tmp_path)
        (tmp_path / "probe_reads.tsv.bak.gapfill").write_text("stale")
        self._run(tmp_path, cores=0, n=100)
        with gzip.open(tmp_path / "probe_reads.tsv.bak.gapfill", "rt") as f:
            assert f.readline().rstrip() == STEP2_HEADER

    @pytest.mark.multiprocess
    def test_multiprocess_matches_single_process(self, tmp_path):
        """Two worker processes write the same file as one."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        _write_reads(tmp_path / "a")
        _write_reads(tmp_path / "b")
        self._run(tmp_path / "a", cores=1)
        self._run(tmp_path / "b", cores=2)
        assert support.read_lines(
            tmp_path / "a" / "probe_reads.tsv.gz"
        ) == support.read_lines(tmp_path / "b" / "probe_reads.tsv.gz")

    def test_missing_output_or_input_raise(self, tmp_path):
        """A missing output directory or reads file raises AssertionError."""
        with pytest.raises(AssertionError, match="Output directory"):
            step3.run(str(tmp_path / "nope"), 1, 1)
        with pytest.raises(AssertionError, match="Input file not found"):
            step3.run(str(tmp_path), 1, 1)

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=UnicodeDecodeError,
        reason="an uncompressed probe_reads.tsv is replaced by gzip-compressed bytes",
    )
    def test_uncompressed_input_stays_readable_as_text(self, tmp_path):
        """An uncompressed `probe_reads.tsv` stays plain text after correction."""
        path = _write_reads(tmp_path, name="probe_reads.tsv")
        self._run(tmp_path)
        assert support.read_lines(path)[0] == STEP3_HEADER


class TestMain:
    def test_arguments_are_forwarded_to_run(self, monkeypatch):
        """Options map to `run(output, cores, n_groups_per_batch)`, defaulting to 1 and 100,000."""
        calls = []
        monkeypatch.setattr(step3, "run", lambda *a: calls.append(a))
        monkeypatch.setattr(
            sys, "argv", ["giftwrap-correct-gapfill", "-o", "out", "-c", "4", "-n", "7"]
        )
        step3.main()
        monkeypatch.setattr(
            sys, "argv", ["giftwrap-correct-gapfill", "--output", "out"]
        )
        step3.main()
        assert calls == [("out", 4, 7), ("out", 1, 100_000)]

    def test_version(self, monkeypatch, capsys):
        """`-v` prints the package version and exits 0."""
        monkeypatch.setattr(sys, "argv", ["giftwrap-correct-gapfill", "-v"])
        with pytest.raises(SystemExit) as exc:
            step3.main()
        assert exc.value.code == 0 and giftwrap.__version__ in capsys.readouterr().out
