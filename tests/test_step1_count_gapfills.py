import gzip
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from Bio.Seq import reverse_complement

import giftwrap
import giftwrap.step1_count_gapfills as step1
import support
from giftwrap.pipeline_utils import (
    FlexFormatInfo,
    FlexV2FormatInfo,
    ProbeParser,
    ReadProcessState as S,
    read_manifest,
    read_manifest_config,
    write_manifest,
)
from giftwrap.step1_count_gapfills import (
    ReadData,
    _expected_gap_length,
    build_manifest,
    collect_unmapped_fastq,
    compute_barcode_read_requirements,
    effective_read_lengths,
    process_reads,
    save_unmapped_data,
    search_files,
    track_read_lengths,
)
from support import (
    CELLS,
    FLEX_BC1,
    FLEX_BC2,
    LHS_A,
    LHS_B,
    LHS_C,
    RHS_A,
    RHS_B,
    UMIS,
    flex_r1,
    flex_r2,
)

RESOURCES = Path(__file__).parents[1] / "src" / "giftwrap" / "resources"
A, B = "GENEA c.1A>T", "GENEB c.2C>G"


def ramp(n):
    return "".join(chr(35 + i % 40) for i in range(n))


def rec(seq, title="r", q=None):
    return (title, seq, q if q is not None else ramp(len(seq)))


def test_read_data_fields():
    """`ReadData` keeps its field names and order."""
    assert ReadData._fields == (
        "probe_id",
        "probe_barcode",
        "gapfill",
        "gapfill_quality",
        "cell_barcode",
        "umi",
        "umi_quality",
        "coordinate_x",
        "coordinate_y",
    )


class TestReadLengths:
    def test_track_read_lengths_passes_pairs_through_and_records_maxima(self):
        """Pairs pass through unchanged while the longest R1 and R2 lengths are recorded."""
        pairs = [
            (rec("AC"), rec("ACGT")),
            (rec("ACGTA"), rec("A")),
            (rec("A"), rec("AC")),
        ]
        lengths = {"R1_CYCLES": 3, "R2_CYCLES": 0}
        assert list(track_read_lengths(iter(pairs), lengths)) == pairs
        assert lengths == {"R1_CYCLES": 5, "R2_CYCLES": 4}

    def test_track_read_lengths_on_empty_input(self):
        """No pairs leave the lengths dict untouched."""
        lengths = {}
        assert list(track_read_lengths(iter([]), lengths)) == []
        assert lengths == {}

    @pytest.mark.parametrize(
        ("observed", "caps", "expected"),
        [
            ({}, (None, None), (0, 0)),
            ({"R1_CYCLES": 100, "R2_CYCLES": 150}, (50, None), (50, 150)),
            ({"R1_CYCLES": 30, "R2_CYCLES": 0}, (50, 90), (30, 90)),
            ({"R1_CYCLES": 28, "R2_CYCLES": 70}, (28, 90), (28, 70)),
        ],
    )
    def test_effective_read_lengths_caps_observed_lengths(
        self, observed, caps, expected
    ):
        """Explicit read lengths cap observed lengths and stand in when nothing was observed."""
        tech = SimpleNamespace(read1_length=caps[0], read2_length=caps[1])
        assert effective_read_lengths(observed, tech) == expected


@pytest.fixture
def uncapped_flex(whitelist):
    return FlexFormatInfo(
        barcode_list=[whitelist], read1_length=None, read2_length=None
    )


@pytest.fixture
def parser(uncapped_flex):
    return ProbeParser([LHS_A, LHS_B], [RHS_A, RHS_B], [A, B], uncapped_flex)


def _process(reads, tech, parser, **kwargs):
    args = (
        dict(
            max_distance=1,
            skip_constant_seq=False,
            unmapped_reads_prefix=None,
            flexible_start=False,
            max_expected_gap=0,
        )
        | kwargs
    )
    return process_reads(reads, tech, parser, **args)


class TestProcessReads:
    def test_exact_read_yields_read_data(self, uncapped_flex, parser):
        """An exact read yields its probe, gapfill and gapfill qualities, cell barcode and UMI."""
        r1, r2 = rec(flex_r1(CELLS[0], UMIS[0])), rec(flex_r2(LHS_A, "TTG", RHS_A))
        [(states, data)] = _process([(r1, r2)], uncapped_flex, parser)
        assert states == [S.TOTAL_READS, S.EXACT]
        assert data == ReadData(
            0, "", "TTG", r2[2][25:28], CELLS[0], UMIS[0], r1[2][16:28], None, None
        )

    def test_corrections_are_reported_instead_of_exact(self, uncapped_flex, parser):
        """Corrected cell barcodes or arms are reported as states instead of EXACT."""
        reads = [
            (
                rec(flex_r1(support.mutate(CELLS[1], 5), UMIS[1])),
                rec(flex_r2(LHS_B, "GCAT", RHS_B)),
            ),
            (
                rec(flex_r1(support.mutate(CELLS[1], 5), UMIS[1])),
                rec(flex_r2(support.mutate(LHS_B, 9), "G", RHS_B)),
            ),
        ]
        (s1, d1), (s2, d2) = _process(reads, uncapped_flex, parser)
        assert (
            s1 == [S.TOTAL_READS, S.CORRECTED_BARCODE] and d1.cell_barcode == CELLS[1]
        )
        assert (
            s2 == [S.TOTAL_READS, S.CORRECTED_LHS, S.CORRECTED_BARCODE]
            and d2.gapfill == "G"
        )

    def test_unknown_cell_barcode_is_filtered(self, uncapped_flex, parser):
        """A cell barcode outside the whitelist filters the read."""
        reads = [(rec("N" * 16 + UMIS[0]), rec(flex_r2(LHS_A, "TTG", RHS_A)))]
        assert _process(reads, uncapped_flex, parser) == [
            ([S.TOTAL_READS, S.FILTERED_NO_CELL_BARCODE], None)
        ]

    def test_probe_failures_are_returned_without_data(self, uncapped_flex, parser):
        """Reads whose probe cannot be parsed return their states but no data."""
        reads = [(rec(flex_r1(CELLS[0], UMIS[0])), rec(flex_r2(LHS_C, "TTG", RHS_A)))]
        assert _process(reads, uncapped_flex, parser) == [
            ([S.TOTAL_READS, S.FILTERED_NO_LHS], None)
        ]

    def test_reads_are_truncated_to_configured_lengths(self, whitelist):
        """Configured read lengths truncate R2 before parsing, here cutting off the constant."""
        tech = FlexFormatInfo(
            barcode_list=[whitelist], read1_length=28, read2_length=40
        )
        pp = ProbeParser([LHS_A], [RHS_A], [A], tech)
        reads = [
            (rec(flex_r1(CELLS[0], UMIS[0], "ACGT")), rec(flex_r2(LHS_A, "TTG", RHS_A)))
        ]
        assert _process(reads, tech, pp) == [
            ([S.TOTAL_READS, S.FILTERED_NO_CONSTANT], None)
        ]

    def test_spatial_reads_carry_spot_coordinates(self, visium_tech):
        """Spatial reads carry the array coordinates of their spot."""
        pp = ProbeParser([LHS_A], [RHS_A], [A], visium_tech)
        reads = [
            (rec(UMIS[0] + "AAACAACGAATAGTTC"), rec(LHS_A + "TTG" + RHS_A + "AAAA"))
        ]
        [(states, data)] = _process(reads, visium_tech, pp)
        assert states == [S.TOTAL_READS, S.EXACT]
        assert (data.cell_barcode, data.umi, data.coordinate_x, data.coordinate_y) == (
            "AAACAACGAATAGTTC",
            UMIS[0],
            17,
            1,
        )

    def test_probe_barcode_read_from_r1(self, whitelist):
        """R1 demultiplexing reads the probe barcode from R1; without one the read is dropped unless the constant is skipped."""
        tech = FlexV2FormatInfo(
            r1_demultiplex=True,
            barcode_list=[whitelist],
            read1_length=None,
            read2_length=None,
        )
        well = tech._index_to_probe_barcodes["A-A02"]
        pp = ProbeParser(
            [LHS_A], [RHS_A], [A], tech, probe_bcs=["A-A02"], r1_demultiplex=True
        )
        r2 = rec(LHS_A + "TTG" + RHS_A + support.FLEX_V2_CONSTANT)
        good = rec(CELLS[0] + UMIS[0] + "T" * 12 + reverse_complement(well) + "AAA")
        bad = rec(CELLS[0] + UMIS[0] + "T" * 12 + "G" * 13)
        (s1, d1), (s2, d2) = _process([(good, r2), (bad, r2)], tech, pp)
        assert d1.probe_barcode == well and s1 == [S.TOTAL_READS, S.EXACT]
        assert d2 is None and s2[-1] == S.FILTERED_NO_PROBE_BARCODE
        [(s3, d3)] = _process([(bad, r2)], tech, pp, skip_constant_seq=True)
        assert d3.probe_barcode is None and d3.gapfill == "TTG"

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="flexible_start gap offsets are relative to the trimmed read, misaligning gapfill_quality"
    )
    def test_flexible_start_gapfill_quality_matches_gapfill_bases(
        self, uncapped_flex, parser
    ):
        """With `flexible_start`, the gapfill quality string belongs to the gapfill bases."""
        r2 = rec("CCA" + flex_r2(LHS_A, "TTG", RHS_A))
        [(_, data)] = _process(
            [(rec(flex_r1(CELLS[0], UMIS[0])), r2)],
            uncapped_flex,
            parser,
            flexible_start=True,
        )
        assert data.gapfill_quality == r2[2][28:31]

    def test_unmapped_reads_are_saved_with_reason(
        self, uncapped_flex, parser, tmp_path
    ):
        """Only unmapped reads are saved, each tagged with the reason it was filtered."""
        prefix = str(tmp_path / "un")
        reads = [
            (
                rec(flex_r1(CELLS[0], UMIS[0]), "a"),
                rec(flex_r2(LHS_C, "T", RHS_A), "a"),
            ),
            (rec("N" * 28, "b"), rec(flex_r2(LHS_A, "TTG", RHS_A), "b")),
            (
                rec(flex_r1(CELLS[0], UMIS[0]), "c"),
                rec(flex_r2(LHS_A, "TTG", RHS_A), "c"),
            ),
        ]
        _process(reads, uncapped_flex, parser, unmapped_reads_prefix=prefix)
        [r1_file] = list((tmp_path / "un_temp" / "R1").iterdir())
        with gzip.open(r1_file, "rt") as f:
            titles = [line for line in f.read().splitlines() if line.startswith("@")]
        assert titles == [
            "@a ReadProcessState.FILTERED_NO_LHS",
            "@b ReadProcessState.FILTERED_NO_CELL_BARCODE",
        ]


class TestUnmappedReads:
    def test_save_is_a_noop_without_prefix(self, tmp_path):
        """Without a prefix, nothing is written."""
        assert (
            save_unmapped_data(None, [(S.FILTERED_NO_LHS, rec("A"), rec("C"))]) is None
        )

    def test_saved_chunks_are_collected_into_paired_fastqs(self, tmp_path):
        """Saved chunks are merged into paired R1/R2 FASTQs and the temp directory is removed."""
        prefix = str(tmp_path / "un")
        save_unmapped_data(
            prefix, [(S.FILTERED_NO_LHS, ("r1", "AC", "II"), ("r1", "GT", "II"))]
        )
        save_unmapped_data(
            prefix, [(S.FILTERED_NO_RHS, ("r2", "A", "I"), ("r2", "C", "I"))]
        )
        assert len(list((tmp_path / "un_temp" / "R2").iterdir())) == 2
        collect_unmapped_fastq(prefix)
        assert not (tmp_path / "un_temp").exists()
        r1 = support.read_lines(tmp_path / "un_R1.fastq.gz")
        r2 = support.read_lines(tmp_path / "un_R2.fastq.gz")
        assert sorted(r1[::4]) == [
            "@r1 ReadProcessState.FILTERED_NO_LHS",
            "@r2 ReadProcessState.FILTERED_NO_RHS",
        ]
        assert r1[::4] == r2[::4]
        assert {r1[1], r1[5]} == {"AC", "A"} and {r2[1], r2[5]} == {"GT", "C"}

    def test_collect_without_prefix_or_chunks_does_nothing(self, tmp_path):
        """Without a prefix or saved chunks, collection writes nothing."""
        assert collect_unmapped_fastq(None) is None
        collect_unmapped_fastq(str(tmp_path / "un"))
        assert list(tmp_path.iterdir()) == []


class TestExpectedGapLength:
    @pytest.mark.parametrize(
        ("gap", "orig", "expected"),
        [
            ("NA", "NA", 0),
            (float("nan"), None, 0),
            ("ACGT", "NA", 4),
            ("NA", "AC", 2),
            ("A", "ACG", 3),
            ("ACG", "", 3),
        ],
    )
    def test_longest_defined_gapfill(self, gap, orig, expected):
        """The expected gap is the longer defined gap sequence, or 0 when both are NA."""
        assert (
            _expected_gap_length(
                {"gap_probe_sequence": gap, "original_gap_probe_sequence": orig}
            )
            == expected
        )


class TestBarcodeReadRequirements:
    def test_r2_barcode_needs_probe_constant_and_barcode(self, flex_tech):
        """An R2 probe barcode needs probe extent + constant + barcode-end cycles of R2."""
        assert compute_barcode_read_requirements(flex_tech, 54) == {
            "min_r1_for_probe_barcode": None,
            "min_r2_for_probe_barcode": 54 + 16 + 10,
        }

    def test_r1_barcode_needs_only_r1_offset(self, whitelist):
        """An R1 probe barcode only needs R1 to reach the end of the barcode."""
        tech = FlexV2FormatInfo(r1_demultiplex=True, barcode_list=[whitelist])
        assert compute_barcode_read_requirements(tech, 54) == {
            "min_r1_for_probe_barcode": 50,
            "min_r2_for_probe_barcode": None,
        }

    def test_technologies_without_barcode_or_layout(self, visium_tech):
        """Technologies without a probe barcode or a defined layout have no requirements."""
        empty = {"min_r1_for_probe_barcode": None, "min_r2_for_probe_barcode": None}
        assert compute_barcode_read_requirements(visium_tech, 54) == empty
        assert (
            compute_barcode_read_requirements(
                SimpleNamespace(has_probe_barcode=True), 54
            )
            == empty
        )


class TestBuildManifest:
    def test_writes_indexed_manifest_and_returns_extents(self, tmp_path):
        """The manifest gets an index column, and the longest probe extent is returned."""
        probes = support.write_probe_file(tmp_path / "probes.tsv")
        assert build_manifest(
            probes, tmp_path / "out", overwrite=False, allow_any_combination=False
        ) == (0, 54)
        df = read_manifest(tmp_path / "out")
        assert df["index"].tolist() == [0, 1]
        assert df["rhs_probe"].tolist() == [RHS_A, RHS_B]
        assert read_manifest_config(tmp_path / "out") == {}

    def test_existing_output_requires_overwrite(self, tmp_path):
        """An existing output directory is replaced only with `overwrite`."""
        probes = support.write_probe_file(tmp_path / "probes.tsv")
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "stale.txt").write_text("x")
        with pytest.raises(AssertionError, match="already exists"):
            build_manifest(probes, tmp_path / "out", False, False)
        build_manifest(probes, tmp_path / "out", True, False)
        assert not (tmp_path / "out" / "stale.txt").exists()

    def test_any_combination_adds_named_decoy_pairs(self, tmp_path, capsys):
        """`allow_any_combination` adds each undefined LHS/RHS pair as an 'LHS/RHS' decoy."""
        probes = support.write_probe_file(tmp_path / "probes.tsv")
        build_manifest(probes, tmp_path / "out", False, True)
        df = read_manifest(tmp_path / "out")
        assert "2 decoy pairings added." in capsys.readouterr().out
        decoys = df[~df.was_defined]
        assert decoys[["lhs_probe", "rhs_probe", "name"]].values.tolist() == [
            [LHS_A, RHS_B, f"{A}/{B}"],
            [LHS_B, RHS_A, f"{B}/{A}"],
        ]
        assert decoys["gene"].isna().all()
        assert df["index"].tolist() == [0, 1, 2, 3]

    def test_trimming_shortens_rhs_to_expected_insert_length(self, tmp_path, capsys):
        """Trimming shortens each RHS to the target length and warns about short or untrimmable probes."""
        probes = support.write_probe_file(tmp_path / "probes.tsv")
        assert build_manifest(
            probes, tmp_path / "out", False, False, trim_probes=40
        ) == (4, 54)
        assert read_manifest(tmp_path / "out")["rhs_probe"].tolist() == [
            RHS_A[:12],
            RHS_B[:11],
        ]
        build_manifest(probes, tmp_path / "out", True, False, trim_probes=33)
        assert "trimmed to less than 5bp" in capsys.readouterr().out
        build_manifest(probes, tmp_path / "out", True, False, trim_probes=20)
        assert "shorter than the trim length" in capsys.readouterr().out
        assert read_manifest(tmp_path / "out")["rhs_probe"].tolist() == [RHS_A, RHS_B]

    def test_config_is_enriched_with_extents_and_read_requirements(
        self, tmp_path, flex_tech
    ):
        """The config header gains probe extents, the expected gap and read-length requirements."""
        probes = support.write_probe_file(tmp_path / "probes.tsv")
        build_manifest(
            probes,
            tmp_path / "a",
            False,
            False,
            trim_probes=40,
            config={"technology": "Flex", "expected_gap_length": -1},
            tech_info=flex_tech,
        )
        assert read_manifest_config(tmp_path / "a") == {
            "technology": "Flex",
            "expected_gap_length": -1,
            "max_probe_extent": 54,
            "max_expected_gap": 4,
            "min_r1_for_probe_barcode": None,
            "min_r2_for_probe_barcode": 80,
        }
        build_manifest(
            probes, tmp_path / "b", False, False, config={"expected_gap_length": 7}
        )
        assert read_manifest_config(tmp_path / "b") == {
            "expected_gap_length": 7,
            "max_probe_extent": 54,
            "max_expected_gap": 7,
        }


def _manifest_dir(tmp_path, name="out"):
    out = tmp_path / name
    out.mkdir()
    write_manifest(out, support.manifest_df())
    return out


SINGLEPLEX_READS = [
    (flex_r1(CELLS[0], UMIS[0]), flex_r2(LHS_A, "TTG", RHS_A)),
    (flex_r1(CELLS[1], UMIS[1]), flex_r2(LHS_B, "GCAT", RHS_B)),
    (flex_r1(CELLS[0], UMIS[2]), flex_r2(LHS_B, "GCAT", RHS_B)),
    (flex_r1(CELLS[0], UMIS[0]), flex_r2(LHS_C, "TTG", RHS_A)),
    ("N" * 16 + UMIS[3], flex_r2(LHS_A, "TTG", RHS_A)),
    (flex_r1(support.mutate(CELLS[0], 3), UMIS[3]), flex_r2(LHS_A, "TAG", RHS_A)),
]


def _search(out, tech, pairs, fq_dir, **kwargs):
    r1, r2 = support.write_fastq_pair(fq_dir, pairs)
    args = dict(cores=1, n_reads_per_batch=2, max_distance=1) | kwargs
    search_files([str(r1)], [str(r2)], out, tech, **args)


def _rows(path):
    return [line.split("\t") for line in support.read_lines(path)]


class TestSearchFiles:
    def test_singleplex_outputs_reads_barcodes_and_metrics(self, tmp_path, flex_tech):
        """A singleplex search writes sorted reads, barcodes in first-seen order and read metrics."""
        out = _manifest_dir(tmp_path)
        _search(out, flex_tech, SINGLEPLEX_READS, tmp_path / "fq")
        reads = _rows(out / "probe_reads.tsv.gz")
        assert reads[0] == [
            "cell_idx",
            "probe_idx",
            "probe_barcode",
            "gapfill",
            "gapfill_quality",
            "umi",
            "umi_quality",
        ]
        assert [(r[0], r[1], r[2], r[3], r[5]) for r in reads[1:]] == [
            ("0", "0", "1", "TTG", UMIS[0]),
            ("0", "0", "1", "TAG", UMIS[3]),
            ("0", "1", "1", "GCAT", UMIS[2]),
            ("1", "1", "1", "GCAT", UMIS[1]),
        ]
        assert support.read_lines(out / "barcodes.tsv.gz") == [
            "barcode\tplex_id\tplex_seq",
            f"{CELLS[0]}-1\t1\t",
            f"{CELLS[1]}-1\t1\t",
        ]
        metrics = (
            pd.read_table(out / "fastq_metrics.tsv")
            .set_index("metric")["value"]
            .to_dict()
        )
        assert metrics == {
            "PROBE_CONTAINING_READS": 4,
            "POSSIBLE_PROBES": 2,
            "PROBES_ENCOUNTERED": 2,
            "R1_CYCLES": 28,
            "R2_CYCLES": 70,
            "TOTAL_READS": 6,
            "EXACT": 3,
            "FILTERED_NO_LHS": 1,
            "FILTERED_NO_CELL_BARCODE": 1,
            "CORRECTED_BARCODE": 1,
        }

    def test_multiplexed_reads_are_labelled_by_probe_barcode(self, tmp_path, flex_tech):
        """Multiplexed reads are labelled by plex, and unknown probe barcodes are dropped."""
        out = _manifest_dir(tmp_path)
        pairs = [
            (flex_r1(CELLS[0], UMIS[0]), flex_r2(LHS_A, "TTG", RHS_A, FLEX_BC1)),
            (flex_r1(CELLS[0], UMIS[1]), flex_r2(LHS_A, "TTG", RHS_A, FLEX_BC2)),
            (flex_r1(CELLS[1], UMIS[2]), flex_r2(LHS_B, "GCAT", RHS_B, "GGGCCCTT")),
        ]
        _search(out, flex_tech, pairs, tmp_path / "fq", multiplex=2)
        assert support.read_lines(out / "barcodes.tsv.gz")[1:] == [
            f"{CELLS[0]}{FLEX_BC1}-1\t1\t{FLEX_BC1}",
            f"{CELLS[0]}{FLEX_BC2}-1\t2\t{FLEX_BC2}",
        ]
        assert [(r[0], r[2]) for r in _rows(out / "probe_reads.tsv.gz")[1:]] == [
            ("0", "1"),
            ("1", "2"),
        ]

    def test_explicit_barcode_ids_are_normalized(self, tmp_path, flex_tech):
        """Barcode ids such as 'BC02' are normalized and only that plex is kept."""
        out = _manifest_dir(tmp_path)
        pairs = [
            (flex_r1(CELLS[0], UMIS[0]), flex_r2(LHS_A, "TTG", RHS_A, FLEX_BC2)),
            (flex_r1(CELLS[0], UMIS[1]), flex_r2(LHS_A, "TTG", RHS_A, FLEX_BC1)),
        ]
        _search(out, flex_tech, pairs, tmp_path / "fq", barcodes=["BC02"])
        assert support.read_lines(out / "barcodes.tsv.gz")[1:] == [
            f"{CELLS[0]}{FLEX_BC2}-1\t2\t{FLEX_BC2}"
        ]

    def test_skipped_constant_assumes_the_single_expected_barcode(
        self, tmp_path, flex_tech
    ):
        """Without a readable probe barcode, the single requested barcode is assumed."""
        out = _manifest_dir(tmp_path)
        pairs = [(flex_r1(CELLS[0], UMIS[0]), LHS_A + "TTG" + RHS_A)]
        _search(
            out,
            flex_tech,
            pairs,
            tmp_path / "fq",
            barcodes=["1"],
            skip_constant_seq=True,
        )
        assert support.read_lines(out / "barcodes.tsv.gz")[1:] == [
            f"{CELLS[0]}{FLEX_BC1}-1\t1\t"
        ]

    @pytest.mark.parametrize(
        ("namer", "label"),
        [(lambda i: f"BC{int(i):02d}", "BC02"), (lambda i: 1 / 0, "2")],
    )
    def test_optional_probe_barcode_names_label_plexes(
        self, tmp_path, flex_tech, namer, label
    ):
        """A technology's `probe_barcode_name` labels plexes; if it raises, the id is used."""
        flex_tech.probe_barcode_name = namer
        flex_tech.make_barcode_string = (
            lambda cell, plex, x, y, multiplexed: f"{cell}-{plex}"
        )
        out = _manifest_dir(tmp_path)
        pairs = [(flex_r1(CELLS[0], UMIS[0]), flex_r2(LHS_A, "TTG", RHS_A, FLEX_BC2))]
        _search(out, flex_tech, pairs, tmp_path / "fq", barcodes=["2"])
        assert support.read_lines(out / "barcodes.tsv.gz")[1:] == [
            f"{CELLS[0]}-{label}\t{label}\t{FLEX_BC2}"
        ]
        assert support.read_lines(out / "probe_reads.tsv.gz")[1].split("\t")[2] == label

    def test_too_many_multiplexed_barcodes_raise(self, tmp_path, flex_tech):
        """Requesting more plexes than the technology defines raises ValueError."""
        out = _manifest_dir(tmp_path)
        with pytest.raises(ValueError, match="defines only 16"):
            _search(out, flex_tech, SINGLEPLEX_READS[:1], tmp_path / "fq", multiplex=17)

    def test_spatial_barcodes_include_tissue_coordinates(self, tmp_path, visium_tech):
        """Spatial runs add in_tissue and array coordinates to the barcodes file."""
        out = _manifest_dir(tmp_path)
        pairs = [(UMIS[0] + "AAACAACGAATAGTTC", LHS_A + "TTG" + RHS_A + "AAAA")]
        _search(out, visium_tech, pairs, tmp_path / "fq")
        assert support.read_lines(out / "barcodes.tsv.gz") == [
            "barcode\tplex_id\tplex_seq\tin_tissue\tarray_col\tarray_row",
            "AAACAACGAATAGTTC-1\t1\t\t1\t17\t1",
        ]

    def test_unmapped_reads_written_next_to_outputs(self, tmp_path, flex_tech):
        """Unmapped reads are written as paired FASTQs in the output directory."""
        out = _manifest_dir(tmp_path)
        _search(
            out,
            flex_tech,
            SINGLEPLEX_READS,
            tmp_path / "fq",
            unmapped_reads_prefix="unmapped",
        )
        assert len(support.read_lines(out / "unmapped_R1.fastq.gz")) == 8
        assert len(support.read_lines(out / "unmapped_R2.fastq.gz")) == 8
        assert not (out / "unmapped_temp").exists()

    @pytest.mark.multiprocess
    def test_multiprocess_matches_single_process(self, tmp_path, flex_tech):
        """Two worker processes produce the same reads, barcodes and metrics as one."""
        a, b = _manifest_dir(tmp_path, "a"), _manifest_dir(tmp_path, "b")
        _search(a, flex_tech, SINGLEPLEX_READS, tmp_path / "fq", cores=1)
        _search(b, flex_tech, SINGLEPLEX_READS, tmp_path / "fq", cores=2)
        for name in ["probe_reads.tsv.gz", "barcodes.tsv.gz"]:
            assert sorted(support.read_lines(a / name)) == sorted(
                support.read_lines(b / name)
            )
        assert (a / "fastq_metrics.tsv").read_text().splitlines()[:6] == (
            b / "fastq_metrics.tsv"
        ).read_text().splitlines()[:6]


class FakeTech:
    def __init__(self, *args):
        self.args = args
        self.n_barcodes = 3
        self.has_probe_barcode = True
        self.probe_barcode_R1 = False

    def normalize_probe_barcode_id(self, bc):
        return f"norm-{bc}"


@pytest.fixture
def run_env(tmp_path, monkeypatch, no_cellranger):
    """Probe file + paired FASTQs with build_manifest and search_files replaced by recorders."""
    calls = {}
    monkeypatch.setattr(
        step1,
        "build_manifest",
        lambda *a, **k: calls.setdefault("build_manifest", (a, k)) and (4, 54),
    )
    monkeypatch.setattr(
        step1, "search_files", lambda *a, **k: calls.setdefault("search_files", (a, k))
    )
    for name in [
        "FlexFormatInfo",
        "FlexV2FormatInfo",
        "VisiumHDFormatInfo",
        "VisiumFormatInfo",
    ]:
        monkeypatch.setattr(step1, name, type(name, (FakeTech,), {}))
    probes = support.write_probe_file(tmp_path / "probes.tsv")
    r1, r2 = support.write_fastq_pair(tmp_path / "fq", SINGLEPLEX_READS[:1])
    return SimpleNamespace(
        calls=calls, probes=str(probes), r1=str(r1), r2=str(r2), tmp=tmp_path
    )


def _run(env, **kwargs):
    args = (
        dict(
            probes=env.probes,
            trim_probes=-1,
            read1=env.r1,
            read2=env.r2,
            project=None,
            output=str(env.tmp / "out"),
            cores=1,
            n_reads_per_batch=10,
            max_distance=1,
            technology="Flex",
            tech_def=None,
            overwrite=False,
            multiplex=1,
            barcodes=None,
            r1_len=None,
            r2_len=None,
            allow_indels=False,
            skip_constant_seq=False,
            allow_any_combination=False,
            unmapped_reads_prefix=None,
            cellranger_output=None,
            flexible_start=False,
            max_expected_gap=-1,
            fallback_sort=False,
        )
        | kwargs
    )
    with pytest.raises(SystemExit) as exc:
        step1.run(step1.RunOptions(**args))
    assert exc.value.code == 0
    return env.calls


class TestRun:
    def test_forwards_configuration_to_manifest_and_search(self, run_env):
        """The run config, technology and search options reach the manifest and search steps."""
        calls = _run(
            run_env,
            barcodes=["BC01"],
            cellranger_output="wta.h5",
            cores=3,
            trim_probes=40,
        )
        (probes, output, overwrite, any_comb, trim), kw = calls["build_manifest"]
        assert (probes, output, trim) == (run_env.probes, run_env.tmp / "out", 40)
        assert kw["config"] == {
            "giftwrap_version": giftwrap.__version__,
            "technology": "Flex",
            "tech_def": None,
            "multiplex": 1,
            "barcodes": ["norm-BC01"],
            "multiplexed": True,
            "skip_constant_seq": False,
            "trim_probes": 40,
            "expected_gap_length": -1,
            "flexible_start_mapping": False,
            "allow_any_combination": False,
            "r1_length": None,
            "r2_length": None,
            "has_probe_barcode": True,
            "probe_barcode_R1": False,
        }
        tech = kw["tech_info"]
        assert tech.args == (RESOURCES, None, None, [Path("wta.h5")])
        (r1s, r2s, out, tech_info), skw = calls["search_files"]
        assert (r1s, r2s, tech_info) == ([run_env.r1], [run_env.r2], tech)
        assert (
            skw["cores"] == 3
            and skw["barcodes"] == ["norm-BC01"]
            and skw["max_expected_gap"] == 4
        )

    def test_explicit_expected_gap_and_cpu_count_for_non_positive_cores(
        self, run_env, monkeypatch
    ):
        """An explicit expected gap overrides the computed one, and zero cores means all CPUs."""
        monkeypatch.setattr(os, "cpu_count", lambda: 5)
        calls = _run(run_env, max_expected_gap=2, cores=0)
        assert calls["search_files"][1]["max_expected_gap"] == 2
        assert calls["search_files"][1]["cores"] == 5

    @pytest.mark.parametrize(
        ("technology", "cls", "first_args"),
        [
            ("Flex", "FlexFormatInfo", ()),
            ("Flex-v2", "FlexV2FormatInfo", (False,)),
            ("Flex-v2-R1", "FlexV2FormatInfo", (True,)),
            ("VisiumHD", "VisiumHDFormatInfo", (None,)),
            ("Visium-v3", "VisiumFormatInfo", (3,)),
        ],
    )
    def test_technology_selects_format_class(
        self, run_env, technology, cls, first_args
    ):
        """Each `--technology` value builds the matching format class with its arguments."""
        tech = _run(run_env, technology=technology, r1_len=28, r2_len=50)[
            "build_manifest"
        ][1]["tech_info"]
        assert type(tech).__name__ == cls
        assert tech.args == (*first_args, RESOURCES, 28, 50, None)

    def test_cellranger_barcode_directory_is_preferred(
        self, run_env, monkeypatch, tmp_path
    ):
        """An installed CellRanger's barcode directory is used when it exists."""
        exe = tmp_path / "cellranger-9.0" / "cellranger"
        (exe.parent / "lib" / "python" / "cellranger" / "barcodes").mkdir(parents=True)
        monkeypatch.setattr(
            shutil, "which", lambda name: str(exe) if name == "cellranger" else None
        )
        tech = _run(run_env)["build_manifest"][1]["tech_info"]
        assert tech.args[0] == exe.parent / "lib" / "python" / "cellranger" / "barcodes"

    def test_spaceranger_without_barcodes_falls_back_to_bundled(
        self, run_env, monkeypatch, tmp_path, capsys
    ):
        """Without a barcode directory next to SpaceRanger, bundled barcodes are used with a warning."""
        exe = tmp_path / "sr" / "spaceranger"
        monkeypatch.setattr(
            shutil, "which", lambda name: str(exe) if name == "spaceranger" else None
        )
        assert _run(run_env)["build_manifest"][1]["tech_info"].args[0] == RESOURCES
        assert "Falling back to default barcodes." in capsys.readouterr().out

    def test_custom_technology_module_on_sys_path(self, run_env, monkeypatch, tmp_path):
        """A custom technology module on `sys.path` is imported and its class instantiated."""
        (tmp_path / "custom_tech_mod.py").write_text(
            "class CustomTech:\n"
            "    n_barcodes = 1\n    has_probe_barcode = False\n    probe_barcode_R1 = False\n"
            "    def __init__(self, *args):\n        self.args = args\n"
            "    def normalize_probe_barcode_id(self, bc):\n        return bc\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "custom_tech_mod", raising=False)
        calls = _run(
            run_env, technology="Custom", tech_def=str(tmp_path / "custom_tech_mod.py")
        )
        tech = calls["build_manifest"][1]["tech_info"]
        assert type(tech).__name__ == "CustomTech" and tech.args == (
            RESOURCES,
            None,
            None,
        )
        assert calls["build_manifest"][1]["config"]["tech_def"] == str(
            tmp_path / "custom_tech_mod.py"
        )

    def test_custom_technology_file_outside_sys_path(self, run_env, tmp_path):
        """A custom technology file is loaded from its path wherever it lives."""
        (tmp_path / "defs").mkdir()
        (tmp_path / "defs" / "other_tech_mod.py").write_text(
            "class T:\n    n_barcodes = 1\n    has_probe_barcode = False\n    probe_barcode_R1 = False\n"
            "    def __init__(self, *a):\n        pass\n"
        )
        _run(
            run_env,
            technology="Custom",
            tech_def=str(tmp_path / "defs" / "other_tech_mod.py"),
        )

    def test_project_prefix_discovers_paired_fastqs(self, run_env, tmp_path):
        """`--project` finds paired R1/R2 FASTQs and ignores non-FASTQ files."""
        support.write_fastq_pair(tmp_path / "proj", SINGLEPLEX_READS[:1], name="sample")
        (tmp_path / "proj" / "sample_R1_notes.txt").write_text("not a fastq")
        calls = _run(
            run_env, read1=None, read2=None, project=str(tmp_path / "proj" / "sample")
        )
        r1s, r2s = calls["search_files"][0][:2]
        assert [Path(p).name for p in r1s] == ["sample_S1_R1_001.fastq.gz"]
        assert [Path(p).name for p in r2s] == ["sample_S1_R2_001.fastq.gz"]

    def test_project_prefix_requires_matching_r2(self, run_env, tmp_path):
        """An R1 file without its R2 partner raises FileNotFoundError."""
        support.write_fastq(tmp_path / "solo_R1_001.fastq.gz", [("r", "A", "I")])
        with pytest.raises(FileNotFoundError, match="Matching R2"):
            step1.run(
                step1.RunOptions(
                    probes=run_env.probes,
                    output=str(tmp_path / "o"),
                    project=str(tmp_path / "solo"),
                )
            )

    def test_relative_read_patterns_pair_files_in_working_directory(
        self, run_env, monkeypatch, tmp_path
    ):
        """Relative R1/R2 prefixes pair the matching lane files in the working directory."""
        (tmp_path / "lanes").mkdir()
        for lane in ("L001", "L002"):
            support.write_fastq(
                tmp_path / "lanes" / f"lib_R1_{lane}.fq", [("r", "A", "I")]
            )
            support.write_fastq(
                tmp_path / "lanes" / f"lib_R2_{lane}.fq", [("r", "C", "I")]
            )
        monkeypatch.chdir(tmp_path / "lanes")
        r1s, r2s = _run(run_env, read1="lib_R1", read2="lib_R2")["search_files"][0][:2]
        assert [Path(p).name for p in r1s] == ["lib_R1_L001.fq", "lib_R1_L002.fq"]
        assert [Path(p).name for p in r2s] == ["lib_R2_L001.fq", "lib_R2_L002.fq"]

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="read patterns with a directory pair each R1 file with itself"
    )
    def test_read_patterns_with_directory_pair_r1_with_r2(self, run_env, tmp_path):
        """R1/R2 prefixes that include a directory pair each R1 with its R2."""
        (tmp_path / "lanes").mkdir()
        support.write_fastq(tmp_path / "lanes" / "lib_R1_L001.fq", [("r", "A", "I")])
        support.write_fastq(tmp_path / "lanes" / "lib_R2_L001.fq", [("r", "C", "I")])
        r1s, r2s = _run(
            run_env,
            read1=str(tmp_path / "lanes" / "lib_R1"),
            read2=str(tmp_path / "lanes" / "lib_R2"),
        )["search_files"][0][:2]
        assert [Path(p).name for p in r2s] == ["lib_R2_L001.fq"]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (dict(read1=None, read2=None, project=None), "At least one"),
            (dict(multiplex=2, barcodes=["1"]), "mutually exclusive"),
            (dict(multiplex=2, skip_constant_seq=True), "only valid for singleplex"),
            (
                dict(barcodes=["1", "2"], skip_constant_seq=True),
                "only valid for singleplex",
            ),
            (dict(read1="missing_R1.fastq.gz"), "Read1 file not found"),
        ],
    )
    def test_invalid_arguments_raise(self, run_env, kwargs, message):
        """Missing or conflicting inputs are rejected with an explanatory AssertionError."""
        with pytest.raises(AssertionError, match=message):
            _run(run_env, **kwargs)

    def test_compression_mismatch_and_missing_probes_raise(self, run_env, tmp_path):
        """Mixed gzip/plain read files or a missing probe file raise AssertionError."""
        plain = support.write_fastq(tmp_path / "x_R2.fastq", [("r", "A", "I")])
        with pytest.raises(AssertionError, match="both be gzipped"):
            _run(run_env, read2=str(plain))
        with pytest.raises(AssertionError, match="Probes file not found"):
            _run(run_env, probes=str(tmp_path / "nope.tsv"))


class TestRunHelpers:
    def test_make_run_config_coerces_options_and_records_technology_flags(self):
        """The run config coerces option types and records the technology's barcode flags."""
        config = step1._make_run_config(
            "Custom",
            Path("defs/t.py"),
            2,
            ["BC01"],
            True,
            40,
            -1,
            False,
            True,
            28,
            None,
            FakeTech(),
        )
        assert config == {
            "giftwrap_version": giftwrap.__version__,
            "technology": "Custom",
            "tech_def": "defs/t.py",
            "multiplex": 2,
            "barcodes": ["BC01"],
            "multiplexed": True,
            "skip_constant_seq": True,
            "trim_probes": 40,
            "expected_gap_length": -1,
            "flexible_start_mapping": False,
            "allow_any_combination": True,
            "r1_length": 28,
            "r2_length": None,
            "has_probe_barcode": True,
            "probe_barcode_R1": False,
        }

    def test_normalize_cellranger_output_accepts_string_list_or_nothing(self):
        """A single path or a list becomes a list of `Path`s, and empty input becomes None."""
        assert step1._normalize_cellranger_output("a.h5") == [Path("a.h5")]
        assert step1._normalize_cellranger_output(["a", "b"]) == [Path("a"), Path("b")]
        assert step1._normalize_cellranger_output([]) is None
        assert step1._normalize_cellranger_output(None) is None

    def test_discover_fastq_pairs_returns_explicit_files_as_one_pair(self, tmp_path):
        """Explicit R1/R2 files are returned as a single pair of path strings."""
        r1, r2 = support.write_fastq_pair(tmp_path / "fq", SINGLEPLEX_READS[:1])
        assert step1._discover_fastq_pairs(str(r1), str(r2), None) == (
            [str(r1)],
            [str(r2)],
        )


class TestLoadCustomTechnology:
    @staticmethod
    def _write(monkeypatch, tmp_path, stem, source):
        monkeypatch.setitem(sys.modules, stem, None)  # the loader registers the module; undone on teardown
        path = tmp_path / f"{stem}.py"
        path.write_text(source)
        return path

    def test_imported_classes_are_not_mistaken_for_the_technology(self, monkeypatch, tmp_path):
        """Classes imported into the definition file are ignored in favour of the class it defines."""
        path = self._write(
            monkeypatch, tmp_path, "ordered_tech",
            "from collections import OrderedDict\n\nclass ZTech:\n    pass\n",
        )
        assert step1._load_custom_technology(path).__name__ == "ZTech"

    def test_technology_subclass_wins_among_several_classes(self, monkeypatch, tmp_path):
        """With several classes defined, the `TechnologyFormatInfo` subclass is selected."""
        path = self._write(
            monkeypatch, tmp_path, "multi_tech",
            "from giftwrap.pipeline_utils import TechnologyFormatInfo\n\n"
            "class Helper:\n    pass\n\nclass MyTech(TechnologyFormatInfo):\n    pass\n",
        )
        assert step1._load_custom_technology(path).__name__ == "MyTech"

    @pytest.mark.parametrize(
        "source",
        ["x = 1\n", "class A:\n    pass\n\nclass B:\n    pass\n"],
        ids=["no-class", "ambiguous"],
    )
    def test_missing_or_ambiguous_class_raises(self, monkeypatch, tmp_path, source):
        """A definition file without exactly one candidate class is rejected with the candidates named."""
        path = self._write(monkeypatch, tmp_path, "bad_tech", source)
        with pytest.raises(ValueError, match="exactly one technology class"):
            step1._load_custom_technology(path)


class TestMain:
    def test_all_arguments_are_forwarded_to_run(self, monkeypatch):
        """Every CLI option reaches `run` in its `RunOptions` field."""
        calls = []
        monkeypatch.setattr(step1, "run", calls.append)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "giftwrap-count",
                "-p",
                "probes.tsv",
                "--trim_probes",
                "40",
                "-r1",
                "R1",
                "-r2",
                "R2",
                "--unmapped_reads",
                "un",
                "--flexible_start_mapping",
                "--project",
                "proj",
                "-o",
                "out",
                "-c",
                "4",
                "-n",
                "100",
                "-t",
                "2",
                "-e",
                "Flex-v2",
                "--tech_def",
                "t.py",
                "-f",
                "-m",
                "1",
                "-b",
                "A01",
                "-b",
                "A02",
                "--r1_length",
                "28",
                "--r2_length",
                "90",
                "--allow_indels",
                "--skip_constant_seq",
                "--allow_any_combination",
                "-wta",
                "a",
                "-wta",
                "b",
                "--expected_gap_length",
                "5",
                "--fallback_sort",
            ],
        )
        step1.main()
        assert calls == [
            step1.RunOptions(
                probes="probes.tsv",
                output="out",
                trim_probes=40,
                read1="R1",
                read2="R2",
                project="proj",
                cores=4,
                n_reads_per_batch=100,
                max_distance=2,
                technology="Flex-v2",
                tech_def="t.py",
                overwrite=True,
                multiplex=1,
                barcodes=["A01", "A02"],
                r1_len=28,
                r2_len=90,
                allow_indels=True,
                skip_constant_seq=True,
                allow_any_combination=True,
                unmapped_reads_prefix="un",
                cellranger_output=["a", "b"],
                flexible_start=True,
                max_expected_gap=5,
                fallback_sort=True,
            )
        ]

    def test_defaults(self, monkeypatch):
        """Only `--probes` and `--output` are required; every other option has a default."""
        calls = []
        monkeypatch.setattr(step1, "run", calls.append)
        monkeypatch.setattr(sys, "argv", ["giftwrap-count", "-p", "p.tsv", "-o", "out"])
        step1.main()
        assert calls == [step1.RunOptions(probes="p.tsv", output="out")]

    def test_rejects_unknown_technology(self, monkeypatch):
        """An unknown `--technology` is a usage error (exit 2)."""
        monkeypatch.setattr(
            sys, "argv", ["giftwrap-count", "-p", "p", "-o", "o", "-e", "Nope"]
        )
        with pytest.raises(SystemExit) as exc:
            step1.main()
        assert exc.value.code == 2

    def test_build_parser_parses_arguments_without_touching_sys_argv(self):
        """`build_parser` returns a standalone parser whose defaults and choices match the CLI."""
        args = step1.build_parser().parse_args(["-p", "p.tsv", "-o", "out", "-e", "Visium-v2"])
        assert (args.probes, args.output, args.technology) == ("p.tsv", "out", "Visium-v2")
        assert (args.cores, args.multiplex, args.n_reads_per_batch) == (1, 1, 10_000_000)
        assert args.barcode is None and args.cellranger_output is None
        with pytest.raises(SystemExit):
            step1.build_parser().parse_args(["-p", "p", "-o", "o", "-e", "Nope"])
