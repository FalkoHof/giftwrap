import gzip
import pickle
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp

import giftwrap.pipeline_utils as pu
import support
from giftwrap.pipeline_utils import (
    FlexFormatInfo,
    FlexV2FormatInfo,
    TechnologyFormatInfo,
    VisiumFormatInfo,
    VisiumHDFormatInfo,
    _normalize_flex_v1_barcode_id,
    _normalize_flex_v2_barcode_id,
    _parse_barcodes_tsv,
    _parse_filtered_feature_bc_matrix_h5,
    _parse_molecule_info_h5,
    _parse_possible_barcodes,
    normalize_barcode_to_target,
    normalize_barcodes_to_target,
    pad_metadata_values,
    read_wta,
)

RESOURCES = Path(__file__).parents[1] / "src" / "giftwrap" / "resources"
SAMPLE_FBM = (
    Path(__file__).parents[1] / "sample_data" / "sample_filtered_feature_bc_matrix.h5"
)
V2_WELLS = pd.read_table(
    RESOURCES / "flex-v2-384.txt", header=None, names=["seq", "corrected", "well"]
)


class TestFlexV1BarcodeIds:
    @pytest.mark.parametrize("raw", ["1", "BC1", "bc01", "BC001", " BC1 ", "01"])
    def test_accepted_spellings_normalize_to_integer_string(self, raw):
        """Plain numbers, 'BC' prefixes, zero padding and surrounding whitespace normalize to '1'."""
        assert _normalize_flex_v1_barcode_id(raw) == "1"

    def test_upper_bound_follows_number_of_barcodes(self):
        """Ids are valid up to `n_barcodes`, which defaults to 16."""
        assert _normalize_flex_v1_barcode_id("BC16") == "16"
        with pytest.raises(ValueError, match="1-4"):
            _normalize_flex_v1_barcode_id("5", n_barcodes=4)

    @pytest.mark.parametrize("raw", ["0", "17", "BCx", "BC-1", "A01", ""])
    def test_invalid_ids_raise(self, raw):
        """Out-of-range or malformed ids raise ValueError."""
        with pytest.raises(ValueError, match="Unknown Flex v1 probe barcode"):
            _normalize_flex_v1_barcode_id(raw)


class TestFlexV2BarcodeIds:
    VALID = V2_WELLS.well.tolist()

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("A-A01", "A-A01"),
            ("a-a01", "A-A01"),
            ("A01", "A-A01"),
            ("a1", "A-A01"),
            ("B-C3", "B-C03"),
            ("D-H12", "D-H12"),
            (" c-b7 ", "C-B07"),
        ],
    )
    def test_accepted_spellings(self, raw, expected):
        """Well ids are uppercased and zero-padded, with plate A as the default."""
        assert _normalize_flex_v2_barcode_id(raw, self.VALID) == expected

    @pytest.mark.parametrize("raw", ["E-A01", "A-I01", "A-A13", "Z", "A-", ""])
    def test_unknown_wells_raise_listing_valid_plates(self, raw):
        """Unknown plates, rows or columns raise ValueError listing the valid plates."""
        with pytest.raises(ValueError, match="valid plates: A, B, C, D"):
            _normalize_flex_v2_barcode_id(raw, self.VALID)


class MiniTech(TechnologyFormatInfo):
    umi_start = property(lambda self: 0)
    umi_length = property(lambda self: 4)
    cell_barcodes = property(lambda self: ["AAAACCCC", "GGGGTTTT"])
    cell_barcode_start = property(lambda self: 4)
    is_spatial = property(lambda self: True)
    barcode_coordinates = property(
        lambda self: {"AAAACCCC": (1, 2), "GGGGTTTT": (3, 4)}
    )
    constant_sequence = property(lambda self: "")
    constant_sequence_start = property(lambda self: 0)
    has_constant_sequence = property(lambda self: False)
    probe_barcodes = property(lambda self: {})
    probe_barcode_start = property(lambda self: 0)
    probe_barcode_length = property(lambda self: 0)
    has_probe_barcode = property(lambda self: False)

    def probe_barcode_index(self, bc):
        raise NotImplementedError


class TestTechnologyFormatInfoBase:
    def test_cannot_instantiate_abstract_base(self):
        """The abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            TechnologyFormatInfo()

    def test_defaults_to_bundled_resources_and_stores_read_lengths(self):
        """The barcode directory defaults to the bundled resources, and read lengths are stored as given."""
        tech = MiniTech(read1_length=28, read2_length=None)
        assert tech._barcode_dir == RESOURCES
        assert tech.read1_length == 28
        assert tech.read2_length is None
        assert MiniTech(barcode_dir="/some/dir")._barcode_dir == Path("/some/dir")

    def test_derived_barcode_helpers(self):
        """Counts, lengths, coordinates, barcode strings and the cached tree derive from subclass properties."""
        tech = MiniTech()
        assert tech.n_barcodes == 2
        assert tech.max_cell_barcode_length == 8
        assert tech.barcode2coordinates("GGGGTTTT") == (3, 4)
        assert tech.probe_barcode_R1 is False
        assert tech.make_barcode_string("AAAACCCC", "2") == "AAAACCCC-2"
        assert tech.barcode_tree is tech.barcode_tree

    def test_correct_barcode_searches_window_with_mismatch_budget(self):
        """The barcode window matches exactly or within the mismatch budget; otherwise (None, -1) is returned."""
        tech = MiniTech()
        assert tech.correct_barcode("UUUUAAAACCCCXX", 1, 4, 12) == ("AAAACCCC", 0)
        assert tech.correct_barcode("UUUUAAAACCCAXX", 1, 4, 12) == ("AAAACCCC", 1)
        assert tech.correct_barcode("UUUUACGTACGTXX", 1, 4, 12) == (None, -1)

    def test_probe_barcode_ids_and_normalization_without_mapping(self):
        """Without a probe barcode map, there are no ids and normalization only strips whitespace."""
        tech = MiniTech()
        assert tech.probe_barcode_ids == []
        assert tech.normalize_probe_barcode_id(" anything ") == "anything"

    def test_normalization_validates_against_known_ids(self):
        """With a probe barcode map, unknown ids raise ValueError listing the valid ones."""
        tech = MiniTech()
        tech._index_to_probe_barcodes = {"1": "AC", "2": "GT"}
        assert tech.probe_barcode_ids == ["1", "2"]
        assert tech.normalize_probe_barcode_id("2 ") == "2"
        with pytest.raises(ValueError, match="Valid ids: 1, 2"):
            tech.normalize_probe_barcode_id("3")


class TestFlexFormatInfo:
    def test_requires_a_barcode_source(self):
        """Without a barcode directory or list, construction raises ValueError."""
        with pytest.raises(ValueError, match="barcode_dir or barcode_list"):
            FlexFormatInfo(barcode_dir=None, barcode_list=None)

    def test_layout_properties(self, flex_tech):
        """Flex v1 uses the 10x layout, with the UMI after the cell barcode and the constant and probe barcode on R2."""
        assert (
            flex_tech.umi_start,
            flex_tech.umi_length,
            flex_tech.cell_barcode_start,
        ) == (16, 12, 0)
        assert flex_tech.is_spatial is False
        assert flex_tech.constant_sequence == support.FLEX_CONSTANT
        assert flex_tech.constant_sequence_start == 0
        assert flex_tech.has_constant_sequence is True
        assert (flex_tech.probe_barcode_start, flex_tech.probe_barcode_length) == (2, 8)
        assert flex_tech.has_probe_barcode is True
        assert flex_tech.probe_barcode_R1 is False
        assert (flex_tech.read1_length, flex_tech.read2_length) == (28, 90)

    def test_whitelist_from_barcode_list_strips_suffix(self, flex_tech):
        """Barcodes from a barcode list lose their '-1' suffix and fill the barcode trie."""
        assert sorted(flex_tech.cell_barcodes) == sorted(support.CELLS)
        assert flex_tech.n_barcodes == len(support.CELLS)
        assert flex_tech.barcode_tree is flex_tech._barcodes
        assert flex_tech.max_cell_barcode_length == 16

    def test_multiplexed_whitelist_is_truncated_to_cell_barcode(self, tmp_path):
        """Multiplexed barcodes in a whitelist are cut to the 16-bp cell barcode."""
        wl = support.write_whitelist(
            tmp_path / "barcodes.tsv.gz", [support.CELLS[0] + support.FLEX_BC1]
        )
        assert FlexFormatInfo(barcode_list=[wl]).cell_barcodes == [support.CELLS[0]]

    def test_probe_barcode_mapping(self, flex_tech):
        """The 16 probe barcodes map to ids 1-16, and ids such as 'BC02' are normalized."""
        assert len(flex_tech.probe_barcodes) == 16
        assert flex_tech.probe_barcode_ids == [str(i) for i in range(1, 17)]
        assert flex_tech.probe_barcode_index(support.FLEX_BC2) == "2"
        assert flex_tech.normalize_probe_barcode_id("BC02") == "2"

    def test_barcode_string_appends_probe_barcode_only_when_multiplexed(
        self, flex_tech
    ):
        """The probe barcode sequence is appended only for multiplexed runs."""
        assert flex_tech.make_barcode_string("CELL", "2") == "CELL-1"
        assert (
            flex_tech.make_barcode_string("CELL", "2", is_multiplexed=True)
            == f"CELL{support.FLEX_BC2}-1"
        )

    def test_coordinates_are_not_available(self, flex_tech):
        """Flex has no spatial coordinates, so accessing them raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            flex_tech.barcode_coordinates

    def test_pickle_round_trip_reattaches_shared_trie(self, flex_tech):
        """Unpickling restores the shared barcode trie, so worker processes can still correct barcodes."""
        restored = pickle.loads(pickle.dumps(flex_tech))
        assert sorted(restored.cell_barcodes) == sorted(support.CELLS)
        assert restored.correct_barcode(support.CELLS[1] + "N" * 12, 2, 0, 16) == (
            support.CELLS[1],
            0,
        )

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=AttributeError,
        reason="FlexFormatInfo.__getstate__ drops the probe barcode maps",
    )
    def test_pickle_round_trip_keeps_probe_barcodes(self, flex_tech):
        """Unpickled objects keep their probe barcode maps."""
        restored = pickle.loads(pickle.dumps(flex_tech))
        assert restored.probe_barcode_index(support.FLEX_BC2) == "2"
        assert restored.probe_barcode_ids == flex_tech.probe_barcode_ids

    @pytest.mark.slow
    def test_bundled_whitelist_is_used_without_barcode_list(self):
        """Without a barcode list, the bundled whitelist of over 700,000 Flex barcodes is used."""
        tech = FlexFormatInfo(barcode_dir=RESOURCES)
        assert tech.n_barcodes > 700_000
        assert all(len(bc) == 16 for bc in tech.cell_barcodes[:10])


class TestFlexV2FormatInfo:
    def test_layout_for_r2_demultiplexing(self, flex_v2_tech):
        """With R2 demultiplexing, the probe barcode starts right after the Flex v2 constant on R2."""
        tech = flex_v2_tech
        assert tech.constant_sequence == support.FLEX_V2_CONSTANT
        assert tech.pCS1() == "CGGTCCTAGCAA"
        assert (tech.probe_barcode_start, tech.probe_barcode_length) == (0, 10)
        assert tech.probe_barcode_R1 is False
        assert (tech.umi_start, tech.umi_length, tech.cell_barcode_start) == (16, 12, 0)
        assert (
            tech.has_constant_sequence
            and tech.has_probe_barcode
            and not tech.is_spatial
        )
        assert tech.constant_sequence_start == 0
        with pytest.raises(NotImplementedError):
            tech.barcode_coordinates

    def test_layout_for_r1_demultiplexing(self, whitelist):
        """With R1 demultiplexing, the probe barcode is read from R1 at offset 40."""
        tech = FlexV2FormatInfo(r1_demultiplex=True, barcode_list=[whitelist])
        assert tech.probe_barcode_start == 40
        assert tech.probe_barcode_R1 is True

    def test_384_well_probe_barcodes(self, flex_v2_tech):
        """The 384 well barcodes map to plate-well ids such as 'A-A01'."""
        assert len(flex_v2_tech.probe_barcodes) == 384
        assert flex_v2_tech.probe_barcode_ids[0] == "A-A01"
        assert flex_v2_tech.probe_barcode_index(V2_WELLS.corrected[0]) == "A-A01"
        assert flex_v2_tech.normalize_probe_barcode_id("a1") == "A-A01"
        assert sorted(flex_v2_tech.cell_barcodes) == sorted(support.CELLS)

    def test_barcode_string_appends_well_sequence_when_multiplexed(self, flex_v2_tech):
        """The well's barcode sequence is appended only for multiplexed runs."""
        assert flex_v2_tech.make_barcode_string("CELL", "A-A01") == "CELL-1"
        assert (
            flex_v2_tech.make_barcode_string("CELL", "A-A01", is_multiplexed=True)
            == f"CELL{V2_WELLS.corrected[0]}-1"
        )

    def test_probe_barcodes_are_found_in_translation_subdirectory(
        self, tmp_path, whitelist
    ):
        """Probe barcodes are also found in a `translation/` subdirectory."""
        (tmp_path / "bcdir" / "translation").mkdir(parents=True)
        shutil.copy(
            RESOURCES / "flex-v2-384.txt",
            tmp_path / "bcdir" / "translation" / "flex-v2-384.txt",
        )
        tech = FlexV2FormatInfo(
            barcode_dir=tmp_path / "bcdir", barcode_list=[whitelist]
        )
        assert len(tech.probe_barcodes) == 384

    def test_pickle_round_trip_keeps_mode_and_probe_barcodes(self, whitelist):
        """Unpickling keeps the R1 demultiplexing mode, the probe barcodes and the cell barcodes."""
        tech = FlexV2FormatInfo(r1_demultiplex=True, barcode_list=[whitelist])
        restored = pickle.loads(pickle.dumps(tech))
        assert restored.probe_barcode_R1 is True
        assert restored.probe_barcode_index(V2_WELLS.corrected[5]) == V2_WELLS.well[5]
        assert sorted(restored.cell_barcodes) == sorted(support.CELLS)

    def test_requires_a_barcode_source(self):
        """Without a barcode directory or list, construction raises ValueError."""
        with pytest.raises(ValueError):
            FlexV2FormatInfo(barcode_dir=None, barcode_list=None)


class TestVisiumFormatInfo:
    @pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
    def test_loads_each_bundled_slide_version(self, version):
        """Each bundled slide version loads every spot with its coordinates."""
        tech = VisiumFormatInfo(version=version, barcode_dir=RESOURCES)
        coords = pd.read_table(
            RESOURCES / f"visium-v{version}_coordinates.txt", header=None
        )
        assert tech.n_barcodes == len(coords)
        first = coords.iloc[0]
        assert tech.barcode2coordinates(first[0]) == (first[1], first[2])

    def test_layout_and_unsupported_features(self, visium_tech):
        """Visium reads start with the UMI, then the spot barcode; constant and probe-barcode features raise NotImplementedError."""
        tech = visium_tech
        assert (tech.umi_start, tech.umi_length, tech.cell_barcode_start) == (0, 12, 12)
        assert (
            tech.is_spatial
            and not tech.has_constant_sequence
            and not tech.has_probe_barcode
        )
        assert tech.make_barcode_string("AAACAACGAATAGTTC", "1") == "AAACAACGAATAGTTC-1"
        assert tech.barcode_coordinates["AAACAACGAATAGTTC"] == (17, 1)
        for attr in [
            "constant_sequence",
            "constant_sequence_start",
            "probe_barcodes",
            "probe_barcode_start",
            "probe_barcode_length",
        ]:
            with pytest.raises(NotImplementedError):
                getattr(tech, attr)
        with pytest.raises(NotImplementedError):
            tech.probe_barcode_index("ACGT")

    def test_corrects_spot_barcode_after_umi(self, visium_tech):
        """A spot barcode with one mismatch after the UMI is corrected."""
        read = "N" * 12 + support.mutate("AAACAACGAATAGTTC", 15)
        assert visium_tech.correct_barcode(read, 2, 12, 28) == ("AAACAACGAATAGTTC", 1)

    def test_requires_a_barcode_source(self):
        """Without a barcode directory or list, construction raises ValueError."""
        with pytest.raises(ValueError):
            VisiumFormatInfo(version=1)

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=AttributeError,
        reason="__getstate__ drops barcode coordinates, so spatial workers fail with --cores > 1",
    )
    def test_pickle_round_trip_keeps_coordinates(self, visium_tech):
        """Unpickled objects keep spot coordinates for worker processes."""
        restored = pickle.loads(pickle.dumps(visium_tech))
        assert restored.barcode2coordinates("AAACAACGAATAGTTC") == (17, 1)

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        raises=KeyError,
        reason="a parsed barcode list is a Series but is indexed like a DataFrame",
    )
    def test_accepts_a_cellranger_barcode_list(self, tmp_path):
        """A SpaceRanger barcode list restricts the spots to those listed."""
        wl = support.write_whitelist(
            tmp_path / "barcodes.tsv.gz", ["AAACAACGAATAGTTC", "AAACAAGTATCTCCCA"]
        )
        tech = VisiumFormatInfo(version=1, barcode_dir=RESOURCES, barcode_list=[wl])
        assert tech.n_barcodes == 2


def _hd_read(offset, bc1, bc2, umi="ACGTTGCAA"):
    return (umi + "GGTTT")[:offset] + bc1 + bc2 + "TTTT"


class TestVisiumHDFormatInfo:
    @pytest.fixture
    def hd(self, fake_spaceranger):
        return VisiumHDFormatInfo(
            space_ranger_path=str(fake_spaceranger["exe"]),
            barcode_dir=fake_spaceranger["barcode_dir"],
        )

    def test_missing_spaceranger_raises(self, monkeypatch, fake_spaceranger):
        """Without SpaceRanger on PATH, construction raises FileNotFoundError."""
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(FileNotFoundError, match="not found on PATH"):
            VisiumHDFormatInfo(barcode_dir=fake_spaceranger["barcode_dir"])

    def test_spaceranger_without_python_libraries_raises(
        self, tmp_path, fake_spaceranger
    ):
        """A SpaceRanger install without its Python libraries raises FileNotFoundError."""
        exe = tmp_path / "bare" / "spaceranger"
        exe.parent.mkdir()
        exe.write_text("")
        with pytest.raises(FileNotFoundError, match="Incorrect spaceranger"):
            VisiumHDFormatInfo(
                space_ranger_path=exe, barcode_dir=fake_spaceranger["barcode_dir"]
            )

    def test_spaceranger_is_discovered_on_path(self, monkeypatch, fake_spaceranger):
        """SpaceRanger is found on PATH when no path is given."""
        monkeypatch.setattr(shutil, "which", lambda name: str(fake_spaceranger["exe"]))
        tech = VisiumHDFormatInfo(barcode_dir=fake_spaceranger["barcode_dir"])
        assert tech.n_barcodes == 6

    def test_chemistry_definitions_from_spaceranger_take_precedence(
        self, fake_spaceranger
    ):
        """SpaceRanger's own chemistry definitions take precedence over the bundled ones."""
        import json

        chem = json.loads(
            (fake_spaceranger["barcode_dir"] / "chemistry_defs.json").read_text()
        )
        chem["SPATIAL-HD-v1"]["barcode"][0]["offset"] = 10
        (
            fake_spaceranger["root"]
            / "lib"
            / "python"
            / "cellranger"
            / "chemistry_defs.json"
        ).write_text(json.dumps(chem))
        tech = VisiumHDFormatInfo(
            space_ranger_path=fake_spaceranger["exe"],
            barcode_dir=fake_spaceranger["barcode_dir"],
        )
        assert tech._default_range == ((10, 24), (25, 39))

    def test_chemistry_definitions_fall_back_to_parent_of_barcode_dir(
        self, tmp_path, fake_spaceranger
    ):
        """Chemistry definitions are also found in the parent of the barcode directory."""
        nested = fake_spaceranger["barcode_dir"] / "nested"
        nested.mkdir()
        (nested / "visium_hd_v1.slide").write_bytes(b"\x00")
        tech = VisiumHDFormatInfo(
            space_ranger_path=fake_spaceranger["exe"], barcode_dir=nested
        )
        assert tech._default_range == ((11, 25), (25, 39))

    def test_missing_chemistry_definitions_raise(self, fake_spaceranger):
        """Without chemistry definitions, construction raises FileNotFoundError."""
        (fake_spaceranger["barcode_dir"] / "chemistry_defs.json").unlink()
        with pytest.raises(FileNotFoundError, match="chemistry_defs.json"):
            VisiumHDFormatInfo(
                space_ranger_path=fake_spaceranger["exe"],
                barcode_dir=fake_spaceranger["barcode_dir"],
            )

    def test_whitelist_coordinates_and_layout(self, hd, fake_spaceranger):
        """The two-part whitelist, bin coordinates, layout and bin names are derived; unsupported features raise."""
        bc1, bc2 = fake_spaceranger["bc1"], fake_spaceranger["bc2"]
        assert hd.n_barcodes == hd.n_cell_barcodes == 6
        assert sorted(hd.cell_barcodes) == sorted(a + b for a in bc1 for b in bc2)
        assert hd.barcode_coordinates[bc1[2] + bc2[1]] == (2, 1)
        assert (hd.umi_start, hd.umi_length, hd.cell_barcode_start) == (0, 9, 9)
        assert (
            hd.is_spatial and not hd.has_constant_sequence and not hd.has_probe_barcode
        )
        assert (hd.min_cell_barcode_length, hd.max_cell_barcode_length) == (14, 18)
        assert hd.get_lengths_to_search == [15, 14, 13]
        assert (
            hd.make_barcode_string("ignored", "1", x_coord=3, y_coord=12)
            == "s_002um_00012_00003-1"
        )
        with pytest.raises(NotImplementedError):
            hd.barcode_tree
        for attr in [
            "constant_sequence",
            "constant_sequence_start",
            "probe_barcodes",
            "probe_barcode_start",
            "probe_barcode_length",
        ]:
            with pytest.raises(NotImplementedError):
                getattr(hd, attr)
        with pytest.raises(NotImplementedError):
            hd.probe_barcode_index("A")

    def test_pairing_ranges_cover_offsets_and_include_default(self, hd):
        """Pairing ranges cover every start offset, always include the default, and grow with padding."""
        ranges = hd.possible_pairing_ranges(43)
        assert ranges == [((o, o + 14), (o + 14, o + 28)) for o in range(8, 13)]
        short = hd.possible_pairing_ranges(30)
        assert ((11, 25), (25, 39)) in short
        assert all(r[1][1] <= 30 for r in short if r != ((11, 25), (25, 39)))
        padded = hd.possible_pairing_ranges(43, padding=1)
        assert len(padded) > len(ranges)

    @pytest.mark.parametrize(
        ("offset", "i1", "i2"), [(11, 0, 1), (9, 2, 0), (12, 1, 1)]
    )
    def test_exact_barcodes_at_shifted_offsets(
        self, hd, fake_spaceranger, offset, i1, i2
    ):
        """Exact barcode pairs are found at shifted start offsets with zero corrections."""
        bc1, bc2 = fake_spaceranger["bc1"][i1], fake_spaceranger["bc2"][i2]
        assert hd.correct_barcode(_hd_read(offset, bc1, bc2), 2, 9, 27) == (
            bc1 + bc2,
            0,
        )

    def test_single_segment_mismatch_is_corrected(self, hd, fake_spaceranger):
        """One mismatch in either barcode segment is corrected and counted once."""
        bc1, bc2 = fake_spaceranger["bc1"][0], fake_spaceranger["bc2"][1]
        assert hd.correct_barcode(
            _hd_read(11, bc1, support.mutate(bc2, 5)), 2, 9, 27
        ) == (bc1 + bc2, 1)
        assert hd.correct_barcode(
            _hd_read(11, support.mutate(bc1, 4), bc2), 2, 9, 27
        ) == (bc1 + bc2, 1)

    def test_both_segments_mismatched_still_recover_barcode(self, hd, fake_spaceranger):
        """Mismatches in both segments still recover the correct barcode."""
        bc1, bc2 = fake_spaceranger["bc1"][0], fake_spaceranger["bc2"][1]
        barcode, _ = hd.correct_barcode(
            _hd_read(11, support.mutate(bc1, 4), support.mutate(bc2, 6)), 2, 9, 27
        )
        assert barcode == bc1 + bc2

    @pytest.mark.known_defect
    @pytest.mark.xfail(
        reason="fuzzy search keeps the candidate with MORE corrections (comparison is inverted)"
    )
    def test_both_segments_mismatched_report_minimal_corrections(
        self, hd, fake_spaceranger
    ):
        """Mismatches in both segments report the minimal number of corrections."""
        bc1, bc2 = fake_spaceranger["bc1"][0], fake_spaceranger["bc2"][1]
        read = _hd_read(11, support.mutate(bc1, 4), support.mutate(bc2, 6))
        assert hd.correct_barcode(read, 2, 9, 27) == (bc1 + bc2, 2)

    def test_unmatched_read_returns_none(self, hd):
        """A read without any known barcode returns (None, -1)."""
        assert hd.correct_barcode("A" * 43, 2, 9, 27) == (None, -1)

    def test_pickle_round_trip(self, hd, fake_spaceranger):
        """Unpickled objects keep their coordinates and still correct barcodes."""
        restored = pickle.loads(pickle.dumps(hd))
        bc1, bc2 = fake_spaceranger["bc1"][1], fake_spaceranger["bc2"][0]
        assert restored.barcode2coordinates(bc1 + bc2) == (1, 0)
        assert restored.correct_barcode(_hd_read(11, bc1, bc2), 2, 9, 27) == (
            bc1 + bc2,
            0,
        )


class TestBarcodeFileParsers:
    @pytest.mark.parametrize("name", ["barcodes.tsv", "barcodes.tsv.gz"])
    def test_tsv_parser_strips_gem_suffix_optionally(self, tmp_path, name):
        """Plain or gzipped barcodes lose only their final '-N' suffix unless `strip_suffix=False`."""
        path = support.write_lines(tmp_path / name, ["AAAA-1", "CC-GG-2"])
        assert _parse_barcodes_tsv(path).tolist() == ["AAAA", "CC-GG"]
        assert _parse_barcodes_tsv(path, strip_suffix=False).tolist() == [
            "AAAA-1",
            "CC-GG-2",
        ]

    def test_molecule_info_parser(self, tmp_path):
        """Barcodes are decoded from `molecule_info.h5`, with optional suffix stripping."""
        with h5py.File(tmp_path / "m.h5", "w") as f:
            f.create_dataset(
                "barcodes", data=np.array([b"s_002um_00001_00002-1", b"BB-1"])
            )
        assert _parse_molecule_info_h5(tmp_path / "m.h5").tolist() == [
            "s_002um_00001_00002",
            "BB",
        ]
        assert (
            _parse_molecule_info_h5(tmp_path / "m.h5", strip_suffix=False).tolist()[1]
            == "BB-1"
        )

    def test_feature_matrix_parser_on_bundled_sample(self):
        """Barcodes are read from a filtered feature matrix, with or without their suffix."""
        full = _parse_filtered_feature_bc_matrix_h5(SAMPLE_FBM, strip_suffix=False)
        stripped = _parse_filtered_feature_bc_matrix_h5(SAMPLE_FBM)
        assert len(full) == len(stripped) > 0
        assert full[0].rsplit("-", 1)[0] == stripped[0]

    def test_possible_barcodes_are_merged_and_deduplicated(self, tmp_path):
        """Barcodes from several sources are merged in order without duplicates."""
        a = support.write_whitelist(tmp_path / "a.tsv.gz", ["AAA", "CCC"])
        b = support.write_whitelist(tmp_path / "b.tsv", ["CCC", "GGG"])
        assert _parse_possible_barcodes([a, b]).tolist() == ["AAA", "CCC", "GGG"]

    def test_possible_barcodes_skip_unparseable_sources(self, tmp_path, capsys):
        """Unparseable sources are skipped with a message, and no usable source gives None."""
        broken = tmp_path / "square_002um"
        broken.mkdir()
        good = support.write_whitelist(tmp_path / "a.tsv.gz", ["AAA"])
        assert _parse_possible_barcodes(
            [broken, tmp_path / "missing.h5", good]
        ).tolist() == ["AAA"]
        assert "Unable to parse barcodes" in capsys.readouterr().out
        assert _parse_possible_barcodes([]) is None
        assert _parse_possible_barcodes([tmp_path / "missing.h5"]) is None


def _write_10x_mtx(directory: Path, barcodes=("AAAA-1", "CCCC-1"), genes=("G1", "G2")):
    directory.mkdir(parents=True, exist_ok=True)
    matrix = sp.coo_matrix(
        np.arange(1, len(genes) * len(barcodes) + 1).reshape(len(genes), len(barcodes))
    )
    scipy.io.mmwrite(directory / "matrix.mtx", matrix)
    with (
        open(directory / "matrix.mtx", "rb") as src,
        gzip.open(directory / "matrix.mtx.gz", "wb") as dst,
    ):
        dst.write(src.read())
    (directory / "matrix.mtx").unlink()
    support.write_lines(
        directory / "features.tsv.gz", [f"ID{g}\t{g}\tGene Expression" for g in genes]
    )
    support.write_lines(directory / "barcodes.tsv.gz", list(barcodes))
    return directory


BARCODE_DIR_LAYOUTS = [
    ("sample_filtered_feature_bc_matrix", ""),
    ("filtered_feature_bc_matrix", ""),
    ("binned_outputs/square_002um/sample_filtered_feature_bc_matrix", "spatial"),
    ("binned_outputs/square_002um/filtered_feature_bc_matrix", "spatial"),
    ("outs/sample_filtered_feature_bc_matrix", ""),
    ("outs/filtered_feature_bc_matrix", ""),
    ("outs/binned_outputs/square_002um/sample_filtered_feature_bc_matrix", ""),
    ("outs/binned_outputs/square_002um/filtered_feature_bc_matrix", ""),
    ("binned_outputs/square_002um/sample_filtered_feature_bc_matrix", ""),
    ("binned_outputs/square_002um/filtered_feature_bc_matrix", ""),
    ("square_002um/sample_filtered_feature_bc_matrix", ""),
    ("square_002um/filtered_feature_bc_matrix", ""),
    (".", ""),
]


class TestReadWtaBarcodesOnly:
    @pytest.mark.parametrize(("layout", "marker"), BARCODE_DIR_LAYOUTS)
    def test_finds_barcodes_in_each_supported_directory_layout(
        self, tmp_path, layout, marker
    ):
        """Barcodes are found in every supported CellRanger and SpaceRanger directory layout."""
        root = tmp_path / "run"
        target = (root / layout).resolve()
        target.mkdir(parents=True, exist_ok=True)
        if marker:
            (root / marker).mkdir()
        support.write_whitelist(target / "barcodes.tsv.gz", ["AAAA", "CCCC"])
        assert read_wta(root, barcodes_only=True).tolist() == ["AAAA", "CCCC"]
        assert read_wta(root, barcodes_only=True, strip_suffix=False).tolist() == [
            "AAAA-1",
            "CCCC-1",
        ]

    @pytest.mark.parametrize(
        "sub", ["sample_filtered_feature_bc_matrix", "filtered_feature_bc_matrix", None]
    )
    def test_binned_square_directory_is_read_directly_or_via_child(self, tmp_path, sub):
        """A binned `square_002um` directory is read directly or through its matrix subdirectory."""
        square = tmp_path / "square_002um"
        target = square / sub if sub else square / "filtered_feature_bc_matrix"
        target.mkdir(parents=True)
        support.write_whitelist(target / "barcodes.tsv.gz", ["AAAA"])
        start = target if sub else square
        assert read_wta(start, barcodes_only=True).tolist() == ["AAAA"]

    def test_binned_square_parent_prefers_sample_matrix(self, tmp_path):
        """In a binned directory, the sample matrix is preferred over the plain one."""
        square = tmp_path / "square_002um"
        support.write_whitelist(
            (square / "sample_filtered_feature_bc_matrix").mkdir(parents=True)
            or square / "sample_filtered_feature_bc_matrix" / "barcodes.tsv.gz",
            ["SSSS"],
        )
        support.write_whitelist(
            (square / "filtered_feature_bc_matrix").mkdir()
            or square / "filtered_feature_bc_matrix" / "barcodes.tsv.gz",
            ["FFFF"],
        )
        assert read_wta(square, barcodes_only=True).tolist() == ["SSSS"]

    def test_unknown_directory_returns_none(self, tmp_path):
        """A directory without known outputs gives None."""
        assert read_wta(tmp_path, barcodes_only=True) is None

    def test_named_h5_files_use_matching_parser(self, tmp_path):
        """Molecule-info and feature-matrix H5 files are parsed according to their names."""
        spatial_run = tmp_path / "hd"
        (spatial_run / "spatial").mkdir(parents=True)
        with h5py.File(spatial_run / "molecule_info.h5", "w") as f:
            f.create_dataset("barcodes", data=np.array([b"MOL-1"]))
        shutil.copy(SAMPLE_FBM, spatial_run / "filtered_feature_bc_matrix.h5")
        shutil.copy(
            spatial_run / "molecule_info.h5", tmp_path / "sample_molecule_info.h5"
        )
        assert read_wta(
            spatial_run / "molecule_info.h5", barcodes_only=True
        ).tolist() == ["MOL"]
        assert read_wta(
            tmp_path / "sample_molecule_info.h5", barcodes_only=True
        ).tolist() == ["MOL"]
        expected = _parse_filtered_feature_bc_matrix_h5(SAMPLE_FBM)
        np.testing.assert_array_equal(
            read_wta(spatial_run / "filtered_feature_bc_matrix.h5", barcodes_only=True),
            expected,
        )
        np.testing.assert_array_equal(
            read_wta(SAMPLE_FBM, barcodes_only=True), expected
        )

    def test_other_files_are_tried_with_each_parser_in_turn(self, tmp_path):
        """Other files are tried with each parser in turn, and unreadable ones give None."""
        tsv = support.write_whitelist(tmp_path / "cells.tsv", ["AAAA"])
        fbm = shutil.copy(SAMPLE_FBM, tmp_path / "wta.h5")
        with h5py.File(tmp_path / "mol.h5", "w") as f:
            f.create_dataset("barcodes", data=np.array([b"MOL-1"]))
        (tmp_path / "junk.bin").write_bytes(b"\x89HDF\x00\x00\xff\xfe")
        assert read_wta(tsv, barcodes_only=True).tolist() == ["AAAA"]
        assert len(read_wta(Path(fbm), barcodes_only=True)) > 0
        assert read_wta(tmp_path / "mol.h5", barcodes_only=True).tolist() == ["MOL"]
        assert read_wta(tmp_path / "junk.bin", barcodes_only=True) is None


ADATA_DIR_LAYOUTS = [
    "filtered_feature_bc_matrix",
    "sample_filtered_feature_bc_matrix",
    "outs/filtered_feature_bc_matrix",
    "outs/sample_filtered_feature_bc_matrix",
    "outs/binned_outputs/square_002um/filtered_feature_bc_matrix",
    "outs/binned_outputs/square_002um/sample_filtered_feature_bc_matrix",
    "binned_outputs/square_002um/filtered_feature_bc_matrix",
    "binned_outputs/square_002um/sample_filtered_feature_bc_matrix",
    "square_002um/filtered_feature_bc_matrix",
    "square_002um/sample_filtered_feature_bc_matrix",
]


class TestReadWtaAnnData:
    @pytest.mark.parametrize("layout", ADATA_DIR_LAYOUTS)
    def test_reads_10x_matrix_directory_layouts(self, tmp_path, layout):
        """10x matrix directories are read as AnnData in every supported layout."""
        _write_10x_mtx(tmp_path / layout)
        adata = read_wta(tmp_path)
        assert adata.obs_names.tolist() == ["AAAA-1", "CCCC-1"]
        assert adata.var_names.tolist() == ["G1", "G2"]
        np.testing.assert_array_equal(adata.X.toarray(), [[1, 3], [2, 4]])

    @pytest.mark.parametrize(
        "name",
        ["filtered_feature_bc_matrix.h5", "sample_filtered_feature_bc_matrix.h5"],
    )
    def test_reads_h5_inside_directory(self, tmp_path, name):
        """A feature-matrix H5 inside a directory is read as AnnData."""
        shutil.copy(SAMPLE_FBM, tmp_path / name)
        adata = read_wta(tmp_path)
        assert adata.n_obs == len(_parse_filtered_feature_bc_matrix_h5(SAMPLE_FBM))

    def test_reads_h5_and_h5ad_files(self, tmp_path):
        """H5 feature matrices and `.h5ad` files are read directly."""
        adata = read_wta(SAMPLE_FBM)
        assert adata.obs_names[0].endswith("-1")
        adata[:3].copy().write_h5ad(tmp_path / "x.h5ad")
        assert read_wta(tmp_path / "x.h5ad").n_obs == 3

    def test_unsupported_or_broken_inputs_return_none(self, tmp_path):
        """Empty directories, unsupported files and corrupt H5 files give None."""
        assert read_wta(tmp_path) is None
        (tmp_path / "x.txt").write_text("hi")
        assert read_wta(tmp_path / "x.txt") is None
        (tmp_path / "bad.h5").write_bytes(b"not hdf5")
        assert read_wta(tmp_path / "bad.h5") is None

    def test_without_scanpy_returns_none_or_falls_back_to_barcodes(
        self, tmp_path, monkeypatch, capsys
    ):
        """Without scanpy, None is returned unless `fallback_to_barcodes` is set."""
        _write_10x_mtx(tmp_path / "filtered_feature_bc_matrix")
        monkeypatch.setitem(sys.modules, "scanpy", None)
        assert read_wta(tmp_path) is None
        assert "Scanpy not found" in capsys.readouterr().out
        assert read_wta(tmp_path, fallback_to_barcodes=True).tolist() == [
            "AAAA",
            "CCCC",
        ]
        assert read_wta(
            tmp_path, fallback_to_barcodes=True, strip_suffix=False
        ).tolist() == ["AAAA-1", "CCCC-1"]


class TestBarcodeNormalization:
    @pytest.mark.parametrize(
        ("barcode", "target", "expected"),
        [
            ("AAAACCCCACTTTAGG-1", "AAAACCCC-1", "AAAACCCC-1"),
            ("AAAACCCC-1", "GGGGTTTT-1", "AAAACCCC-1"),
            ("AAAACCCCGG-2", "GGGGTTTT-1", "AAAACCCC-2"),
            ("AAAACCCC-1", "GGGGTTTT", "AAAACCCC"),
            ("AAAACCCC", "GGGGTTTT-1", "AAAACCCC"),
            ("AAAACCCCGG", "GGGGTTTT", "AAAACCCC"),
        ],
    )
    def test_single_barcode(self, barcode, target, expected):
        """Barcodes are cut to the target's length and keep their own '-N' suffix only if the target has one."""
        assert normalize_barcode_to_target(barcode, target) == expected

    def test_vectorized_handles_mixed_formats(self):
        """Vectorized normalization handles barcodes with and without suffixes together."""
        out = normalize_barcodes_to_target(
            ["AAAACCCCGG-1", "TTTTGGGG-1", "CCCCAAAA"], "GGGGTTTT-1"
        )
        assert out.tolist() == ["AAAACCCC-1", "TTTTGGGG-1", "CCCCAAAA"]

    def test_vectorized_drops_suffix_for_suffixless_target(self):
        """Suffixes are dropped when the target has none."""
        assert normalize_barcodes_to_target(["AAAACCCCGG-1"], "GGGGTTTT").tolist() == [
            "AAAACCCC"
        ]

    def test_vectorized_returns_input_for_empty_inputs(self):
        """An empty barcode list or an empty target returns the input unchanged."""
        assert normalize_barcodes_to_target([], "GGGG-1").tolist() == []
        assert normalize_barcodes_to_target(["AAAACC-1"], "").tolist() == ["AAAACC-1"]


class TestPadMetadataValues:
    def test_integer_columns_are_padded_with_minus_one_keeping_dtype(self):
        """Integer columns are padded with -1 and keep their dtype."""
        out = pad_metadata_values(pd.Series([1, 2], dtype=np.int32), 3)
        assert out.tolist() == [-1, -1, -1]
        assert out.dtype == np.int32

    def test_float_columns_are_padded_with_nan(self):
        """Float columns are padded with NaN."""
        out = pad_metadata_values(pd.Series([1.5]), 2)
        assert np.isnan(out).all() and out.dtype == np.float64

    def test_byte_and_text_columns_are_padded_with_empty_values(self):
        """Byte, text and empty object columns are padded with empty values."""
        assert pad_metadata_values(pd.Series([b"x"]), 2).tolist() == [b"", b""]
        assert pad_metadata_values(pd.Series(["x"]), 1).tolist() == [""]
        assert pad_metadata_values(pd.Series([], dtype=object), 2).tolist() == ["", ""]
