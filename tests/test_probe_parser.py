import pickle

import pytest
from Bio.Seq import reverse_complement

import support
from giftwrap.pipeline_utils import FlexV2FormatInfo, ProbeParser, ReadProcessState as S
from support import FLEX_BC2, FLEX_CONSTANT, LHS_A, LHS_B, LHS_C, RHS_A, RHS_B, flex_r2

FAILED = (None, None, None, None, None)


@pytest.fixture
def parser(flex_tech):
    return ProbeParser([LHS_A, LHS_B], [RHS_A, RHS_B], ["a", "b"], flex_tech)


@pytest.fixture
def bc_parser(flex_tech):
    return ProbeParser(
        [LHS_A, LHS_B], [RHS_A, RHS_B], ["a", "b"], flex_tech, probe_bcs=["1", "2"]
    )


class TestConstruction:
    def test_deduplicates_arms_and_records_lengths(self, flex_tech):
        """Shared arms map to all their probe indices; arm lengths are listed longest first."""
        pp = ProbeParser(
            [LHS_A, LHS_A, LHS_B[:20]],
            [RHS_A, RHS_B, RHS_B],
            ["a", "b", "c"],
            flex_tech,
        )
        assert dict(pp.deduped_lhs) == {LHS_A: {0, 1}, LHS_B[:20]: {2}}
        assert dict(pp.deduped_rhs) == {RHS_A: {0}, RHS_B: {1, 2}}
        assert pp.lhs_lens == [25, 20] and pp.lhs_lens_differ
        assert pp.rhs_lens == [25] and not pp.rhs_lens_differ

    def test_constant_and_probe_barcode_tries_follow_technology(
        self, flex_tech, visium_tech
    ):
        """Constant and probe-barcode tries are built only when the technology uses them."""
        flex = ProbeParser([LHS_A], [RHS_A], ["a"], flex_tech)
        assert (
            flex.has_constant_seq
            and flex.constant_seq == FLEX_CONSTANT
            and flex.constant_seq_start == 0
        )
        assert not flex.has_probe_bc and flex.probe_bc_start is None
        with_bc = ProbeParser([LHS_A], [RHS_A], ["a"], flex_tech, probe_bcs=[2])
        assert with_bc.has_probe_bc and (
            with_bc.probe_bc_start,
            with_bc.probe_bc_length,
        ) == (2, 8)
        assert not ProbeParser(
            [LHS_A], [RHS_A], ["a"], flex_tech, probe_bcs=[]
        ).has_probe_bc
        visium = ProbeParser([LHS_A], [RHS_A], ["a"], visium_tech, probe_bcs=["1"])
        assert (
            not visium.has_constant_seq
            and visium.constant_seq is None
            and not visium.has_probe_bc
        )

    @pytest.mark.parametrize(
        ("length", "per_10bp", "expected"),
        [(25, 2, 5), (5, 1, 1), (16, 1, 1), (30, 1, 3)],
    )
    def test_distance_budget_scales_with_length(
        self, parser, length, per_10bp, expected
    ):
        """The mismatch budget is length * per-10bp budget // 10, at least 1."""
        assert parser._compute_max_distance(length, per_10bp) == expected


class TestParseProbe:
    def test_exact_read_returns_probe_gapfill_and_positions(self, parser):
        """An exact read returns the probe index, gapfill and gapfill offsets."""
        assert parser.parse_probe(flex_r2(LHS_A, "TTG", RHS_A), 1) == (
            0,
            "TTG",
            25,
            28,
            "",
            [S.TOTAL_READS],
        )
        assert parser.parse_probe(flex_r2(LHS_B, "", RHS_B), 1) == (
            1,
            "",
            25,
            25,
            "",
            [S.TOTAL_READS],
        )

    def test_mismatched_arms_are_corrected_and_flagged(self, parser):
        """A mismatch in either arm is corrected and flagged with the matching state."""
        lhs = parser.parse_probe(flex_r2(support.mutate(LHS_A, 3), "TTG", RHS_A), 1)
        rhs = parser.parse_probe(flex_r2(LHS_A, "TTG", support.mutate(RHS_A, 10)), 1)
        assert lhs[:2] == (0, "TTG") and lhs[-1] == [S.TOTAL_READS, S.CORRECTED_LHS]
        assert rhs[:2] == (0, "TTG") and rhs[-1] == [S.TOTAL_READS, S.CORRECTED_RHS]

    @pytest.mark.parametrize(
        ("read", "state"),
        [
            (flex_r2(LHS_C, "TTG", RHS_A), S.FILTERED_NO_LHS),
            (flex_r2(LHS_A, "TTG", LHS_C), S.FILTERED_NO_RHS),
            (flex_r2(LHS_A, "TTG", RHS_B), S.FILTERED_NO_RHS),
            (LHS_A + "TTG" + RHS_A + "GGTTCCAAGGTTCCAA", S.FILTERED_NO_CONSTANT),
        ],
    )
    def test_unmappable_reads_report_the_failing_component(self, parser, read, state):
        """An unmappable read reports which component failed to match."""
        assert parser.parse_probe(read, 1) == (*FAILED, [S.TOTAL_READS, state])

    def test_skip_constant_sequence_maps_reads_without_it(self, parser):
        """With `skip_constant_seq`, reads without the constant sequence still map."""
        read = LHS_A + "TTG" + RHS_A + "GGTTCCAAGGTTCCAA"
        assert parser.parse_probe(read, 1, True) == (
            0,
            "TTG",
            25,
            28,
            "",
            [S.TOTAL_READS],
        )

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="longest_prefix_match only matches complete trie entries, so a cut-off constant is never rescued"
    )
    def test_constant_sequence_cut_off_after_half_its_length_is_tolerated(self, parser):
        """A read whose constant sequence is cut off after half its length still maps."""
        read = LHS_A + "TTG" + RHS_A + FLEX_CONSTANT[:10]
        assert parser.parse_probe(read, 1)[:2] == (0, "TTG")

    def test_short_read_path_maps_complete_rhs_with_expected_gap(self, parser):
        """Reads shorter than the RHS plus expected gap use the prefix search and still map."""
        read = LHS_A + "TTG" + RHS_A
        assert parser.parse_probe(read, 1, True, False, 10) == (
            0,
            "TTG",
            25,
            28,
            "",
            [S.TOTAL_READS],
        )

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="the truncated-RHS prefix search cannot match a partial RHS (prefix matcher needs full entries)"
    )
    def test_short_read_with_truncated_rhs_is_mapped(self, parser):
        """A short read whose RHS is cut off still maps to its probe."""
        read = LHS_A + "TTG" + RHS_A[:15]
        assert parser.parse_probe(read, 1, True, False, 10)[:2] == (0, "TTG")

    def test_variable_length_lhs_tries_longest_first_and_skips_too_long(
        self, flex_tech
    ):
        """With mixed LHS lengths, the longest matching LHS wins and LHSs longer than the read are skipped."""
        pp = ProbeParser([LHS_A, LHS_B[:20]], [RHS_A, RHS_B], ["a", "b"], flex_tech)
        assert pp.parse_probe(flex_r2(LHS_B[:20], "GG", RHS_B), 1)[:4] == (
            1,
            "GG",
            20,
            22,
        )
        assert pp.parse_probe(flex_r2(LHS_A, "GG", RHS_A), 1)[:2] == (0, "GG")
        assert pp.parse_probe(LHS_B[:20] + "T", 1)[-1] == [
            S.TOTAL_READS,
            S.FILTERED_NO_CONSTANT,
        ]
        assert pp.parse_probe("ACGT", 1)[-1] == [S.TOTAL_READS, S.FILTERED_NO_LHS]
        corrected = pp.parse_probe(
            flex_r2(support.mutate(LHS_B[:20], 2), "GG", RHS_B), 1
        )
        assert corrected[0] == 1 and S.CORRECTED_LHS in corrected[-1]

    def test_shared_lhs_is_disambiguated_by_rhs(self, flex_tech):
        """A LHS shared by two probes is resolved by the RHS that matches."""
        pp = ProbeParser([LHS_A, LHS_A], [RHS_A, RHS_B], ["a", "b"], flex_tech)
        assert pp.parse_probe(flex_r2(LHS_A, "C", RHS_B), 1)[0] == 1

    def test_flexible_start_finds_lhs_after_an_insertion(self, parser):
        """With `flexible_start`, the LHS is found after leading inserted bases."""
        result = parser.parse_probe(
            "CCA" + flex_r2(LHS_A, "TTG", RHS_A), 1, False, True
        )
        assert result[:2] == (0, "TTG")
        assert result[-1] == [S.TOTAL_READS]
        corrected = parser.parse_probe(
            "CCA" + flex_r2(support.mutate(LHS_A, 12), "TTG", RHS_A), 1, False, True
        )
        assert corrected[0] == 0 and corrected[-1] == [S.TOTAL_READS, S.CORRECTED_LHS]
        assert parser.parse_probe("CCA" + LHS_C * 2, 1, False, True)[-1] == [
            S.TOTAL_READS,
            S.FILTERED_NO_LHS,
        ]

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="flexible_start returns gap offsets relative to the trimmed read, not the original R2"
    )
    def test_flexible_start_gap_positions_index_the_original_read(self, parser):
        """Gap offsets index the original read even when bases precede the LHS."""
        read = "CCA" + flex_r2(LHS_A, "TTG", RHS_A)
        _, gap, start, end, _, _ = parser.parse_probe(read, 1, False, True)
        assert read[start:end] == gap == "TTG"

    def test_technology_without_constant_uses_whole_read(self, visium_tech):
        """Without a constant sequence, the RHS is searched across the rest of the read."""
        pp = ProbeParser([LHS_A], [RHS_A], ["a"], visium_tech)
        assert pp.parse_probe(LHS_A + "GAT" + RHS_A + "ACGTACGT", 1) == (
            0,
            "GAT",
            25,
            28,
            "",
            [S.TOTAL_READS],
        )

    def test_results_are_cached_per_parser(self, parser):
        """Parsing the same read again returns the cached result object."""
        read = flex_r2(LHS_A, "TTG", RHS_A)
        assert parser.parse_probe(read, 1) is parser.parse_probe(read, 1)


class TestProbeBarcodes:
    def test_r2_probe_barcode_is_returned(self, bc_parser):
        """The probe barcode two bases after the R2 constant sequence is returned."""
        assert bc_parser.parse_probe(flex_r2(LHS_A, "TTG", RHS_A, FLEX_BC2), 1) == (
            0,
            "TTG",
            25,
            28,
            FLEX_BC2,
            [S.TOTAL_READS],
        )

    def test_corrected_r2_probe_barcode(self, bc_parser):
        """A probe barcode with one mismatch is corrected."""
        assert (
            bc_parser.parse_probe(
                flex_r2(LHS_A, "TTG", RHS_A, support.mutate(FLEX_BC2, 3)), 1
            )[4]
            == FLEX_BC2
        )

    @pytest.mark.parametrize("probe_bc", [None, "GGGCCCTT", FLEX_BC2[:5]])
    def test_missing_unknown_or_truncated_probe_barcode_is_filtered(
        self, bc_parser, probe_bc
    ):
        """Missing, unknown or truncated probe barcodes filter the read."""
        read = flex_r2(LHS_A, "TTG", RHS_A, probe_bc)
        assert bc_parser.parse_probe(read, 1) == (
            *FAILED,
            [S.TOTAL_READS, S.FILTERED_NO_PROBE_BARCODE],
        )


def _v2_r1(well_seq: str, shift: int = 0) -> str:
    return (
        support.CELLS[0]
        + support.UMIS[0]
        + "T" * (12 + shift)
        + reverse_complement(well_seq)
        + "AAA"
    )


class TestProbeBarcodeOnR1:
    @pytest.fixture
    def r1_parser(self, whitelist):
        tech = FlexV2FormatInfo(r1_demultiplex=True, barcode_list=[whitelist])
        self.well = tech._index_to_probe_barcodes["A-A02"]
        return ProbeParser(
            [LHS_A], [RHS_A], ["a"], tech, probe_bcs=["A-A02"], r1_demultiplex=True
        )

    @pytest.mark.parametrize("shift", [-1, 0, 2])
    def test_reverse_complemented_barcode_is_found_near_expected_offset(
        self, r1_parser, shift
    ):
        """The reverse-complemented R1 barcode is found when shifted -1 to +3 bases."""
        assert r1_parser.parse_probe_bc_R1(_v2_r1(self.well, shift), 1) == (
            self.well,
            [],
        )

    def test_read_too_short_for_barcode_window(self, r1_parser):
        """An R1 read that ends before the barcode window is filtered."""
        assert r1_parser.parse_probe_bc_R1("ACGT" * 9, 1) == (
            None,
            [S.FILTERED_NO_PROBE_BARCODE],
        )

    def test_unknown_barcode(self, r1_parser):
        """An R1 barcode outside the requested set is filtered."""
        assert r1_parser.parse_probe_bc_R1(_v2_r1("GGGGGGGGGG"), 1) == (
            None,
            [S.FILTERED_NO_PROBE_BARCODE],
        )

    def test_r2_barcode_is_not_parsed_when_demultiplexing_from_r1(self, r1_parser):
        """When demultiplexing from R1, no probe barcode is read from R2."""
        read = LHS_A + "TTG" + RHS_A + support.FLEX_V2_CONSTANT + "GGGGGGGGGG"
        assert r1_parser.parse_probe(read, 1) == (0, "TTG", 25, 28, "", [S.TOTAL_READS])


def test_pickle_round_trip_reloads_shared_tries(bc_parser):
    """An unpickled parser reattaches its shared tries and parses identically."""
    restored = pickle.loads(pickle.dumps(bc_parser))
    read = flex_r2(LHS_B, "GCAT", RHS_B, FLEX_BC2)
    assert restored.parse_probe(read, 1) == bc_parser.parse_probe(read, 1)
    assert restored.has_probe_bc and restored.names == ["a", "b"]
