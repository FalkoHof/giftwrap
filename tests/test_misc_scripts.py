import hashlib
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

import giftwrap
import giftwrap.misc_scripts as misc
import support
from giftwrap.misc_scripts import int_to_hash
from giftwrap.pipeline_utils import FlexFormatInfo

R_SCRIPT = Path(giftwrap.__file__).parent / "read_gf_h5.R"


def _cli(monkeypatch, func, *args):
    monkeypatch.setattr(sys, "argv", [func.__name__, *args])
    return func()


class TestPrintR:
    def test_prints_bundled_reader_script_and_exits(self, monkeypatch, capsys):
        """`giftwrap-generate-r` prints the bundled R reader verbatim and exits 0."""
        with pytest.raises(SystemExit) as exc:
            _cli(monkeypatch, misc.print_R)
        assert exc.value.code == 0
        assert capsys.readouterr().out == R_SCRIPT.read_text()

    def test_version_flag(self, monkeypatch, capsys):
        """`--version` prints the package version and exits 0."""
        with pytest.raises(SystemExit) as exc:
            _cli(monkeypatch, misc.print_R, "--version")
        assert exc.value.code == 0 and giftwrap.__version__ in capsys.readouterr().out


class TestPrintTech:
    def test_prints_flex_definition_as_template(self, monkeypatch, capsys):
        """`giftwrap-generate-tech` prints an import line and the FlexFormatInfo source."""
        with pytest.raises(SystemExit) as exc:
            _cli(monkeypatch, misc.print_tech)
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert out.startswith("from giftwrap import")
        assert out.endswith(inspect.getsource(FlexFormatInfo))

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=ImportError,
        reason="the printed template imports names giftwrap does not export",
    )
    def test_printed_template_is_importable(self, monkeypatch, capsys):
        """The printed template runs as-is, so users can start from it directly."""
        with pytest.raises(SystemExit):
            _cli(monkeypatch, misc.print_tech)
        exec(compile(capsys.readouterr().out, "template.py", "exec"), {})


class TestIntToHash:
    @pytest.mark.parametrize(
        ("n", "raw"), [(0, b"\x00"), (255, b"\xff"), (256, b"\x01\x00")]
    )
    def test_sha1_prefix_of_big_endian_bytes(self, n, raw):
        """The id is the SHA-1 prefix of the integer's minimal big-endian bytes."""
        assert int_to_hash(n) == hashlib.sha1(raw).hexdigest()[:7]

    def test_length_parameter_and_distinct_outputs(self):
        """`length` sets the id size, and consecutive integers get distinct ids."""
        assert len(int_to_hash(5, length=12)) == 12
        assert len({int_to_hash(i) for i in range(200)}) == 200


def _revert(monkeypatch, capsys, path):
    _cli(monkeypatch, misc.revert_probes, "--input", str(path))
    return capsys.readouterr().out.splitlines()


class TestRevertProbes:
    HEADER = [
        "#probe_set_file_format=3.0",
        "#panel_name=GIFTWRAP probe set",
        "#panel_type=predesigned",
        "#reference_genome=GRCh38",
        "#reference_version=2024-A",
        "gene_id,probe_seq,probe_id,included,region,gene_name",
    ]

    def test_gapfill_panel_yields_ref_and_alt_probes(
        self, tmp_path, monkeypatch, capsys
    ):
        """Each REF/ALT gapfill probe becomes a REF and an ALT CellRanger probe of <=50 bp."""
        lines = _revert(
            monkeypatch, capsys, support.write_probe_file(tmp_path / "p.tsv")
        )
        assert lines[:6] == self.HEADER
        name = "GENEA c.1A>T"
        assert lines[6] == (
            f"{name} REF,{(support.LHS_A + 'TAG' + support.RHS_A)[:50]},"
            f"{name} REF|GENEA|{int_to_hash(0)},TRUE,unspliced,GENEA REF"
        )
        assert lines[7].endswith(
            f",{(support.LHS_A + 'TTG' + support.RHS_A)[:50]},"
            f"{name} ALT|GENEA|{int_to_hash(1)},TRUE,unspliced,GENEA ALT"
        )
        assert len(lines) == 6 + 4

    def test_panel_without_gapfills_yields_one_probe_each_with_gene_from_name(
        self, tmp_path, monkeypatch, capsys
    ):
        """Probes without gapfills map one-to-one; the gene is the first word of the name."""
        df = pd.DataFrame(
            {
                "name": ["GENEX probe1"],
                "lhs_probe": [support.LHS_A],
                "rhs_probe": [support.RHS_A],
            }
        )
        lines = _revert(
            monkeypatch, capsys, support.write_probe_file(tmp_path / "p.csv", df)
        )
        assert lines[6:] == [
            f"GENEX probe1,{(support.LHS_A + support.RHS_A)[:50]},"
            f"GENEX probe1|GENEX|{int_to_hash(0)},TRUE,unspliced,GENEX"
        ]

    def test_missing_input_raises(self, tmp_path, monkeypatch):
        """A missing `--input` file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _cli(monkeypatch, misc.revert_probes, "--input", str(tmp_path / "nope.tsv"))

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="the ALT gene_id is printed with a doubled space ('name  ALT')"
    )
    def test_alt_gene_id_uses_single_space(self, tmp_path, monkeypatch, capsys):
        """ALT probes are named '<name> ALT', matching the REF naming."""
        lines = _revert(
            monkeypatch, capsys, support.write_probe_file(tmp_path / "p.tsv")
        )
        assert lines[7].startswith("GENEA c.1A>T ALT,")

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=TypeError,
        reason="probes without gap sequences crash when any probe has REF/ALT gaps",
    )
    def test_mixed_panel_with_control_probes(self, tmp_path, monkeypatch, capsys):
        """Controls without gap sequences can be mixed with REF/ALT probes in one panel."""
        path = support.write_probe_file(
            tmp_path / "p.tsv", support.probe_panel_df(include_control=True)
        )
        assert len(_revert(monkeypatch, capsys, path)) == 6 + 5


class TestConvertProbes:
    CSV = [
        "#probe_set_file_format=3.0",
        "gene_id,probe_seq,probe_id,included,region",
        f"ENSG1,{support.LHS_A}{support.RHS_A},ENSG1|GENEA|abc,TRUE,spliced",
        f"ENSG2,{support.LHS_B}{support.RHS_B},ENSG2|GENEB|def,TRUE,unspliced",
    ]

    def test_splits_cellranger_probes_into_arms(self, tmp_path, monkeypatch, capsys):
        """Each 50-bp probe is split into 25-bp LHS/RHS arms with the gene from the probe id."""
        path = support.write_lines(tmp_path / "probes.csv", self.CSV)
        _cli(monkeypatch, misc.convert_probes, "--input", str(path))
        lines = capsys.readouterr().out.splitlines()
        assert lines == [
            "name\tlhs_probe\trhs_probe\tgene\tgap_probe_sequence\toriginal_gap_probe_sequence",
            f"ENSG1|GENEA|abc\t{support.LHS_A}\t{support.RHS_A}\tGENEA\t\t",
            f"ENSG2|GENEB|def\t{support.LHS_B}\t{support.RHS_B}\tGENEB\t\t",
        ]

    def test_without_region_column(self, tmp_path, monkeypatch, capsys):
        """Probe sets without a `region` column convert too."""
        csv = [
            "gene_id,probe_seq,probe_id,included",
            f"E,{support.LHS_A}{support.RHS_A},E|G|x,TRUE",
        ]
        _cli(
            monkeypatch,
            misc.convert_probes,
            "--input",
            str(support.write_lines(tmp_path / "p.csv", csv)),
        )
        assert capsys.readouterr().out.splitlines()[1].startswith("E|G|x\t")

    def test_defaults_to_the_published_human_probe_set(
        self, tmp_path, monkeypatch, capsys
    ):
        """Without `--input`, the 10x human WTA v1.0.1 probe set is fetched with a browser user agent."""
        seen = {}
        frame = pd.read_csv(
            support.write_lines(tmp_path / "p.csv", self.CSV), comment="#"
        )

        def fake_read_csv(source, **kwargs):
            seen["source"], seen["kwargs"] = source, kwargs
            return frame.copy()

        monkeypatch.setattr(misc.pd, "read_csv", fake_read_csv)
        _cli(monkeypatch, misc.convert_probes)
        assert seen["source"].startswith("https://cf.10xgenomics.com/") and seen[
            "source"
        ].endswith("v1.0.1_GRCh38-2020-A.csv")
        assert (
            seen["kwargs"]["comment"] == "#"
            and "User-Agent" in seen["kwargs"]["storage_options"]
        )
        assert len(capsys.readouterr().out.splitlines()) == 3
