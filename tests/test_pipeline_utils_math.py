import numpy as np
import pytest
import scipy.sparse as sp

from giftwrap.pipeline_utils import (
    compute_max_distance,
    generate_permuted_seqs,
    interpret_phred_letter,
    permute_bases,
    phred_string_to_probs,
    sequence_saturation_curve,
    sequencing_saturation,
)


@pytest.mark.parametrize(
    ("seq_len", "per_10bp", "expected"),
    [
        (0, 2, 1),
        (9, 2, 2),
        (10, 2, 2),
        (11, 2, 4),
        (25, 1, 3),
        (16, 0, 1),
        (100, 3, 30),
    ],
)
def test_compute_max_distance_rounds_length_up_to_next_10bp_with_floor_of_one(
    seq_len, per_10bp, expected
):
    """The edit budget is the per-10bp budget times the length rounded up to 10bp, at least 1."""
    assert compute_max_distance(seq_len, per_10bp) == expected


class TestPhred:
    @pytest.mark.parametrize(
        ("letter", "expected"), [("I", 1e-4), ("+", 0.1), ("5", 10**-2), ("!", 1.0)]
    )
    def test_default_offset_33_converts_letter_to_error_probability(
        self, letter, expected
    ):
        """With the default Illumina base, letters use the Phred+33 offset: P = 10^(-Q/10)."""
        assert interpret_phred_letter(letter) == pytest.approx(expected)

    def test_phred64_offset_is_applied_for_non_default_base(self):
        """Any base other than 'illumina' applies the Phred+64 offset ('h' is Q40)."""
        assert interpret_phred_letter("h", base="sanger") == pytest.approx(1e-4)

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="'sanger' applies the Phred+64 offset; Sanger FASTQ quality is Phred+33"
    )
    def test_sanger_quality_uses_phred33_offset(self):
        """Sanger-encoded qualities use the Phred+33 offset ('I' is Q40)."""
        assert interpret_phred_letter("I", base="sanger") == pytest.approx(1e-4)

    @pytest.mark.parametrize("bad", ["", "II"])
    def test_rejects_anything_but_a_single_character(self, bad):
        """Empty or multi-character quality strings are rejected."""
        with pytest.raises(AssertionError):
            interpret_phred_letter(bad)

    def test_string_conversion_maps_each_character(self):
        """Every character of a quality string is converted; an empty string gives []."""
        assert phred_string_to_probs("I+!") == pytest.approx([1e-4, 0.1, 1.0])
        assert phred_string_to_probs("") == []


class TestPermutations:
    def test_permute_bases_enumerates_all_bases_at_given_positions_only(self):
        """All 4^k base combinations are produced at the listed positions; others are untouched."""
        out = list(permute_bases("AAA", [0, 2]))
        assert len(out) == 16
        assert len(set(out)) == 16
        assert out[0] == "AAA"
        assert all(s[1] == "A" for s in out)
        assert {s[0] for s in out} == set("ACGT")

    def test_permute_bases_without_positions_yields_input(self):
        """With no positions to permute, the input sequence is yielded once."""
        assert list(permute_bases("ACG", [])) == ["ACG"]

    def test_generate_permuted_seqs_visits_positions_by_descending_quality_score(self):
        """Positions are permuted in descending order of their quality score."""
        quality = np.array([0.1, 0.9, 0.5])
        out = list(generate_permuted_seqs("AAA", quality, 1))
        assert len(out) == 3 * 4
        assert out[:4] == ["AAA", "ACA", "AGA", "ATA"]
        assert out[4:8] == ["AAA", "AAC", "AAG", "AAT"]

    def test_generate_permuted_seqs_count_for_two_edits(self):
        """Two edits over four positions yield P(4, 2) * 4^2 sequences."""
        out = list(generate_permuted_seqs("ACGT", np.array([1.0, 2.0, 3.0, 4.0]), 2))
        assert len(out) == 4 * 3 * 16


class TestSaturation:
    def test_saturation_is_one_minus_unique_over_total_reads(self):
        """Saturation is 1 - (non-zero entries / total reads)."""
        assert sequencing_saturation(np.array([[1, 2], [3, 0]])) == pytest.approx(0.5)

    def test_saturation_is_zero_without_duplicates(self):
        """Exactly one read per entry means zero saturation."""
        assert sequencing_saturation(np.ones((3, 3), dtype=int)) == 0.0

    def test_saturation_accepts_sparse_input(self):
        """Sparse count matrices give the same saturation as dense ones."""
        assert sequencing_saturation(sp.csr_matrix([[1, 2], [3, 0]])) == pytest.approx(
            0.5
        )

    def test_saturation_of_empty_counts_is_nan(self):
        """An all-zero matrix has undefined (NaN) saturation."""
        with np.errstate(invalid="ignore", divide="ignore"):
            assert np.isnan(sequencing_saturation(np.zeros((2, 2), dtype=int)))

    def test_curve_shape_and_full_depth_endpoint(self):
        """At full depth the curve reaches the observed saturation and mean reads per cell."""
        counts = sp.csr_matrix(np.array([[4, 0, 1], [2, 3, 0]]))
        with np.errstate(invalid="ignore", divide="ignore"):
            curve = sequence_saturation_curve(counts, n_points=3)
        assert curve.shape == (3, 2)
        assert curve[-1, 0] == pytest.approx(5.0)
        assert curve[-1, 1] == pytest.approx(sequencing_saturation(counts.toarray()))
        assert curve[0, 0] <= curve[-1, 0]
