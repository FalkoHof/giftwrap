# GIFTwrap test suite

Tests for the `giftwrap` package (`src/giftwrap`). For how to install the dependencies and run the suite, see the
"Running the Tests" section of `docs/contributing.md`. In short:

```bash
uv sync --all-extras --all-groups
uv run pytest
```

pytest is configured in `pyproject.toml` (`[tool.pytest.ini_options]`): tests are collected from `tests/`, `tests/` is
added to `sys.path` so `support.py` can be imported, unknown markers are errors, and expected failures are strict.

## Layout

Each module in `src/giftwrap` has a matching test module.

| Test module | Covers |
|---|---|
| `test_package_init.py` | `giftwrap/__init__.py`, `giftwrap/analysis/__init__.py` (version, exports, NumPy aliases) |
| `test_analysis_utils.py` | `analysis_utils.py`: pool helpers, sparse HDF5 reading, spatial-coordinate parsing and repair, `read_h5_file` |
| `test_pipeline_utils_io.py` | `pipeline_utils.py` I/O: manifest and run-config headers, pipeline invocation history, gzip helpers, probe-file parsing, FASTQ reading, TSV sorting, `compile_flatfile` |
| `test_pipeline_utils_barcodes.py` | `pipeline_utils.py` technologies and barcodes: the Flex, Flex v2, Visium and Visium HD format classes, barcode-ID normalization, `read_wta` and the barcode-file parsers |
| `test_probe_parser.py` | `pipeline_utils.ProbeParser`: mapping of R2 reads to probes, gapfills and probe barcodes |
| `test_h5_filters.py` | `pipeline_utils.py` count-file filters: by cell barcode and by PCR duplicates; `real_gapfill_mask` |
| `test_pipeline_utils_math.py` | `pipeline_utils.py` numeric helpers: edit-distance budgets, Phred conversion, base permutations, sequencing saturation |
| `test_step1_count_gapfills.py` | `step1_count_gapfills.py` (`giftwrap-count`) |
| `test_step2_correct_umis.py` | `step2_correct_umis.py` (`giftwrap-correct-umis`) |
| `test_step3_correct_gapfill.py` | `step3_correct_gapfill.py` (`giftwrap-correct-gapfill`) |
| `test_step4_collect_counts.py` | `step4_collect_counts.py` (`giftwrap-collect`) |
| `test_step5_summarize_counts.py` | `step5_summarize_counts.py` (`giftwrap-summarize`), including the PDF report |
| `test_html_report.py` | `html_report.py`: every figure and QC-flag helper, plus the full HTML report |
| `test_pipeline_cli.py` | `pipeline.py` (`giftwrap`): the orchestrator that runs the five steps |
| `test_misc_scripts.py` | `misc_scripts.py`: `giftwrap-generate-r`, `giftwrap-generate-tech`, `giftwrap-convert-probes`, `giftwrap-revert-probes` |
| `test_analysis_tools.py` | `analysis/tools.py` (`gw.tl`): genotype calling, annotation, imputation, connectivity, off-by-one correction, motif alignment |
| `test_analysis_preprocess.py` | `analysis/preprocess.py` (`gw.pp`) |
| `test_analysis_plots.py` | `analysis/plots.py` (`gw.pl`) |
| `test_analysis_spatial.py` | `analysis/spatial.py` (`gw.sp`): binning, joining with WTA `SpatialData`, plotting, imputation, co-clustering |
| `test_end_to_end.py` | Runs steps 1–5 on a tiny synthetic Flex library, singleplex and multiplexed, and checks the final counts |

## Shared helpers

### `support.py`
Plain functions and constants used to build small, fully known inputs. Import them with `import support`.

- **Sequences:** `LHS_A`/`RHS_A`, `LHS_B`/`RHS_B` (two 25-bp probe pairs), `LHS_C` (an arm that is not in the panel), `CELLS` (16-bp cell barcodes), `UMIS` (12-bp UMIs), `FLEX_CONSTANT`, `FLEX_V2_CONSTANT`, `FLEX_BC1`/`FLEX_BC2` (Flex v1 probe barcodes). The sequences are chosen to be far apart, so a single substitution from `mutate()` is always corrected to the intended sequence.
- **Reads:** `flex_r1(cell, umi)` and `flex_r2(lhs, gap, rhs, probe_bc=None)` build Flex read layouts; `write_fastq` and `write_fastq_pair` write them, gzipped or plain.
- **Panels:** `probe_panel_df()` and `write_probe_file()` (CSV, TSV or XLSX) give a two-probe gapfill panel, optionally with a `0bp` control; `manifest_df()` adds the manifest `index` column.
- **Pipeline files:** `write_tsv`, `write_lines`, `read_lines` (gzip chosen by suffix), `write_whitelist` (a CellRanger-style `barcodes.tsv.gz`), `write_10x_mtx` (a `filtered_feature_bc_matrix` folder).
- **Count files:** `write_counts_h5(...)` writes a counts HDF5 file with exactly the layout produced by `giftwrap-collect`. Optional metadata, cell and probe indices, and PCR-threshold layers can be supplied.
- **AnnData:** `make_gapfill_adata(...)` builds an object shaped like `read_h5_file` output (var index `probe|gapfill`, `total_reads` and `percent_supporting` layers), sparse or dense. `to_dense()` converts either kind of matrix for assertions.

### `conftest.py` fixtures

| Fixture | Provides |
|---|---|
| *(autouse)* | Seeds NumPy's global RNG and closes matplotlib figures after every test; matplotlib uses the `Agg` backend. |
| `whitelist` | A `barcodes.tsv.gz` containing `support.CELLS`. |
| `flex_tech`, `flex_v2_tech` | Flex and Flex v2 technology objects built from `whitelist`, avoiding the 737K-barcode bundled lists. |
| `visium_tech` | A Visium v1 technology object using the bundled spot coordinates. |
| `no_cellranger` | Makes `shutil.which` report that `cellranger` and `spaceranger` are not installed, so step 1 falls back to bundled barcodes. |
| `fake_spaceranger` | A minimal SpaceRanger install with a stub slide-design protobuf (three `bc1` and two `bc2` oligos), so `VisiumHDFormatInfo` can be tested without SpaceRanger. |

## Conventions

- **Synthetic data with exact expectations.** Inputs are a handful of reads, cells or probes, so expected matrices and statistics can be written out in full. Prefer asserting exact values over shapes or "not empty".
- **Real code paths; mock only at boundaries.** Dependencies run for real. Mocks are used only where real execution is impossible or unsafe in a test: launching pipeline subprocesses (`pipeline.main`), network downloads (`giftwrap-convert-probes` without `--input`), SpaceRanger, the undeclared `spatialdata_io` package, and CellRanger discovery.
- **Isolation.** Write files under `tmp_path` and patch global state (`sys.argv`, `os.cpu_count`, `shutil.which`, `sys.modules`) with `monkeypatch`, so tests never touch the repository or each other.
- **Command-line entry points.** Each `main()` is tested by setting `sys.argv` and replacing the module's `run` with a recorder. The `run()` functions call `exit(0)`, so assert `pytest.raises(SystemExit)` with `code == 0`.
- **Docstrings.** Every test has a one-line docstring in the present tense that states the expected behavior and adds detail the name leaves out, such as thresholds, fallbacks or edge cases. For `known_defect` tests it describes the correct behavior; the `xfail` reason describes the bug.
- **Markers** (declared in `pyproject.toml`):
  - `slow`: loads large bundled resources or renders heavy plots.
  - `multiprocess`: uses a real worker pool (`cores=2`).
  - `integration`: runs several pipeline steps end to end.
  - `known_defect`: documents a confirmed bug (see below).

## Known defects

A confirmed bug that is not fixed yet gets a test asserting the *correct* behavior, marked like this:

```python
@pytest.mark.known_defect
@pytest.mark.xfail(raises=ValueError, reason="max() over an empty mapping when a plex has no valid reads")
def test_plex_without_reads_produces_an_empty_count_file(tmp_path):
    ...
```

- The `reason` states what is wrong. Use `raises=` when the bug shows up as a specific exception, so an unrelated error is still reported as a failure.
- Expected failures are strict, so once the bug is fixed the test is reported as `XPASS(strict)` and the run fails. Remove both markers when fixing the bug; the test then guards against regressions.
- List all current entries with `uv run pytest -m known_defect -rx`.

Some known defects need extra care:

- **pyFAMSA profile alignment** (`_compute_alignments` with at least two motifs in both groups, and the off-by-one redistribution path that relies on it) aborts the Python interpreter. These tests therefore run the code in a subprocess and assert on its exit code.
- **Lazy single-core results:** with `cores=1`, `maybe_multiprocess` returns an `itertools.starmap` that is evaluated lazily. Tests that cover parallel code paths use `cores=2` (marker `multiprocess`); the `cores=1` failures are covered by their own `known_defect` tests.

## Adding tests

1. Put the test in the module that matches the source file, inside a `Test<Function>` class when a function needs several cases.
2. Name tests after the behavior they check, e.g. `test_missing_requested_cells_are_padded_with_zero_rows`, and add a one-line docstring (see Conventions).
3. Build inputs with `support.py` and the fixtures above; add a helper to `support.py` only if two or more modules need it.
4. Run the module you changed, then the whole suite, and confirm it ends with only `passed` and `xfailed`.
