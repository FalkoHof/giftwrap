import json
import shutil
import sys
from pathlib import Path

import pytest

import giftwrap
import giftwrap.pipeline as pipeline
import giftwrap.pipeline_utils as pu
from giftwrap.pipeline import streaming_subprocess_run

STEPS = [
    "giftwrap-count",
    "giftwrap-correct-umis",
    "giftwrap-correct-gapfill",
    "giftwrap-collect",
    "giftwrap-summarize",
]
MARKERS = [
    "COUNT_GAPFILLS",
    "CORRECT_UMIS",
    "CORRECT_GAPFILLS",
    "COLLECT_COUNTS",
    "ANALYSIS",
]


class TestStreamingSubprocessRun:
    def test_streams_merged_output_and_returns_exit_code(self, capsys):
        """The command is echoed, stdout and stderr are streamed together, and its exit code is returned."""
        code = "import sys; print('out1'); print('err1', file=sys.stderr, flush=True); print('out2'); sys.exit(3)"
        assert streaming_subprocess_run([sys.executable, "-c", code]) == 3
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == f"> {sys.executable} -c {code}"
        assert sorted(lines[1:]) == ["err1", "out1", "out2"]

    def test_successful_command_and_keyword_arguments(self, tmp_path, capsys):
        """Keyword arguments such as `cwd` are passed through to the subprocess."""
        assert (
            streaming_subprocess_run(
                [sys.executable, "-c", "import os; print(os.getcwd())"], cwd=tmp_path
            )
            == 0
        )
        assert capsys.readouterr().out.splitlines()[-1] == str(tmp_path)


@pytest.fixture
def fake_steps(monkeypatch):
    """Replace subprocess execution with a recorder; ``codes`` maps step name -> exit code."""
    state = {"calls": [], "codes": {}}

    def run(args, **kwargs):
        name = Path(args[0]).name
        state["calls"].append(args)
        out = Path(args[args.index("-o") + 1])
        if name == "giftwrap-count":
            if "--overwrite" in args and out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
        return state["codes"].get(name, 0)

    monkeypatch.setattr(pipeline, "streaming_subprocess_run", run)
    return state


def _main(monkeypatch, *args):
    argv = ["giftwrap", *args]
    monkeypatch.setattr(sys, "argv", argv)
    pipeline.main()
    return argv


class TestPipelineMain:
    def test_runs_all_steps_with_minimal_arguments(
        self, tmp_path, monkeypatch, fake_steps, capsys
    ):
        """Minimal arguments run the five step commands in order and mark each step done."""
        out = str(tmp_path / "out")
        _main(
            monkeypatch, "-p", "probes.tsv", "-o", out, "-r1", "R1.fq", "-r2", "R2.fq"
        )
        calls = fake_steps["calls"]
        assert [Path(c[0]).name for c in calls] == STEPS
        assert all(c[0] == sys.path[0] + "/" + Path(c[0]).name for c in calls)
        assert calls[0][1:] == [
            "-p",
            "probes.tsv",
            "-o",
            out,
            "-c",
            "1",
            "-e",
            "Flex",
            "-r1",
            "R1.fq",
            "-r2",
            "R2.fq",
        ]
        assert calls[1][1:] == ["-o", out, "-c", "1"]
        assert calls[2][1:] == ["-o", out, "-c", "1"]
        assert calls[3][1:] == ["-o", out, "-c", "1", "--max_pcr_thresholds", "10"]
        assert calls[4][1:] == ["-o", out]
        assert sorted(p.name for p in (tmp_path / "out" / "steps").iterdir()) == sorted(
            MARKERS
        )
        captured = capsys.readouterr()
        assert "Gapfill counts pipeline completed." in captured.out
        assert f"GIFTwrap Pipeline Version: {giftwrap.__version__}" in captured.out

    def test_records_the_invocation_after_counting(
        self, tmp_path, monkeypatch, fake_steps
    ):
        """The orchestrator records its command line in the output directory."""
        argv = _main(
            monkeypatch,
            "-p",
            "p.tsv",
            "-o",
            str(tmp_path / "out"),
            "-r1",
            "a",
            "-r2",
            "b",
        )
        history = json.loads(
            (tmp_path / "out" / pu.PIPELINE_INVOCATION_FILE).read_text()
        )
        assert [h["argv"] for h in history] == [argv]

    def test_all_options_are_forwarded_to_the_right_steps(
        self, tmp_path, monkeypatch, fake_steps
    ):
        """Every option is forwarded only to the steps that use it."""
        out = str(tmp_path / "out")
        _main(
            monkeypatch,
            "-p",
            "p.tsv",
            "-o",
            out,
            "--project",
            "proj",
            "-c",
            "8",
            "-e",
            "Flex-v2",
            "--trim_probes",
            "40",
            "--flexible_start_mapping",
            "--tech_def",
            "t.py",
            "-f",
            "--r1_length",
            "28",
            "--r2_length",
            "90",
            "-m",
            "2",
            "--skip_constant_seq",
            "--allow_any_combination",
            "--unmapped_reads",
            "un",
            "--expected_gap_length",
            "0",
            "--fallback_sort",
            "-wta",
            "a",
            "-wta",
            "b",
            "-ac",
            "--flatten",
            "--reads_per_gapfill",
            "3",
            "--max_pcr_thresholds",
            "4",
        )
        count, umis, gapfill, collect, summarize = fake_steps["calls"]
        assert count[1:] == [
            "-p",
            "p.tsv",
            "-o",
            out,
            "-c",
            "8",
            "-e",
            "Flex-v2",
            "--trim_probes",
            "40",
            "--flexible_start_mapping",
            "--tech_def",
            "t.py",
            "--overwrite",
            "--project",
            "proj",
            "--r1_length",
            "28",
            "--r2_length",
            "90",
            "-m",
            "2",
            "--skip_constant_seq",
            "--allow_any_combination",
            "--unmapped_reads",
            "un",
            "--expected_gap_length",
            "0",
            "--fallback_sort",
            "-wta",
            "a",
            "-wta",
            "b",
        ]
        assert umis[1:] == ["-o", out, "-c", "8", "--allow_chimeras"]
        assert gapfill[1:] == ["-o", out, "-c", "8"]
        assert collect[1:] == [
            "-o",
            out,
            "-c",
            "8",
            "--max_pcr_thresholds",
            "4",
            "--multiplex",
            "--overwrite",
            "--flatten",
        ]
        assert summarize[1:] == [
            "-o",
            out,
            "--overwrite",
            "--flatten",
            "--reads_per_gapfill",
            "3",
            "-wta",
            "a",
            "-wta",
            "b",
        ]

    def test_explicit_barcodes_are_repeated(self, tmp_path, monkeypatch, fake_steps):
        """Each `-b` barcode is passed to counting, and collection is not told to multiplex."""
        _main(
            monkeypatch,
            "-p",
            "p",
            "-o",
            str(tmp_path / "o"),
            "-r1",
            "a",
            "-r2",
            "b",
            "-b",
            "A01",
            "-b",
            "A02",
        )
        count = fake_steps["calls"][0]
        assert count[-4:] == ["-b", "A01", "-b", "A02"]
        assert "--multiplex" not in fake_steps["calls"][3]

    def test_completed_steps_are_skipped(
        self, tmp_path, monkeypatch, fake_steps, capsys
    ):
        """Steps whose marker file exists are skipped and reported as done."""
        steps = tmp_path / "out" / "steps"
        steps.mkdir(parents=True)
        for marker in MARKERS[:3]:
            (steps / marker).touch()
        _main(
            monkeypatch, "-p", "p", "-o", str(tmp_path / "out"), "-r1", "a", "-r2", "b"
        )
        assert [Path(c[0]).name for c in fake_steps["calls"]] == STEPS[3:]
        err = capsys.readouterr().err
        assert (
            "Step 1: Gapfill counting already done." in err
            and "Step 3: Gapfill correction already done." in err
        )

    def test_fully_completed_run_executes_nothing(
        self, tmp_path, monkeypatch, fake_steps, capsys
    ):
        """With every marker present, no step runs."""
        steps = tmp_path / "out" / "steps"
        steps.mkdir(parents=True)
        for marker in MARKERS:
            (steps / marker).touch()
        _main(
            monkeypatch, "-p", "p", "-o", str(tmp_path / "out"), "-r1", "a", "-r2", "b"
        )
        assert fake_steps["calls"] == []
        assert "Summary statistics and plots already done." in capsys.readouterr().err

    def test_overwrite_reruns_completed_steps(self, tmp_path, monkeypatch, fake_steps):
        """`--overwrite` reruns every step despite existing markers."""
        steps = tmp_path / "out" / "steps"
        steps.mkdir(parents=True)
        for marker in MARKERS:
            (steps / marker).touch()
        _main(
            monkeypatch,
            "-p",
            "p",
            "-o",
            str(tmp_path / "out"),
            "-r1",
            "a",
            "-r2",
            "b",
            "--overwrite",
        )
        assert [Path(c[0]).name for c in fake_steps["calls"]] == STEPS

    @pytest.mark.parametrize("failing", range(5))
    def test_a_failing_step_stops_the_pipeline(
        self, tmp_path, monkeypatch, fake_steps, failing, capsys
    ):
        """A failing step exits 1, runs no later step and writes no marker for itself."""
        fake_steps["codes"][STEPS[failing]] = 2
        with pytest.raises(SystemExit) as exc:
            _main(
                monkeypatch,
                "-p",
                "p",
                "-o",
                str(tmp_path / "out"),
                "-r1",
                "a",
                "-r2",
                "b",
            )
        assert exc.value.code == 1
        assert [Path(c[0]).name for c in fake_steps["calls"]] == STEPS[: failing + 1]
        steps = tmp_path / "out" / "steps"
        done = sorted(p.name for p in steps.iterdir()) if steps.exists() else []
        assert done == sorted(MARKERS[:failing])
        assert "Error: Failed to" in capsys.readouterr().err

    def test_invocation_recording_failures_only_warn(
        self, tmp_path, monkeypatch, fake_steps, capsys
    ):
        """Failing to record the invocation only warns; all steps still run."""

        def boom(*args):
            raise OSError("read-only")

        monkeypatch.setattr(pu, "record_pipeline_invocation", boom)
        _main(
            monkeypatch, "-p", "p", "-o", str(tmp_path / "out"), "-r1", "a", "-r2", "b"
        )
        assert (
            "could not record the pipeline invocation: read-only"
            in capsys.readouterr().err
        )
        assert len(fake_steps["calls"]) == 5

    def test_multiplex_and_barcode_are_mutually_exclusive(
        self, tmp_path, monkeypatch, fake_steps
    ):
        """`--multiplex` with `--barcode` is a usage error raised before any step runs."""
        with pytest.raises(SystemExit) as exc:
            _main(monkeypatch, "-p", "p", "-o", str(tmp_path), "-m", "2", "-b", "A01")
        assert exc.value.code == 2
        assert fake_steps["calls"] == []

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="missing -r1/-r2/--project is not validated; None is forwarded into the step command"
    )
    def test_missing_read_inputs_are_rejected_before_running_steps(
        self, tmp_path, monkeypatch, fake_steps
    ):
        """Without reads or a project, the CLI exits before running any step."""
        with pytest.raises(SystemExit):
            _main(monkeypatch, "-p", "p", "-o", str(tmp_path / "out"))

    def test_version(self, monkeypatch, capsys):
        """`--version` prints the package version and exits 0."""
        monkeypatch.setattr(sys, "argv", ["giftwrap", "--version"])
        with pytest.raises(SystemExit) as exc:
            pipeline.main()
        assert exc.value.code == 0 and giftwrap.__version__ in capsys.readouterr().out
