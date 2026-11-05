"""Builders shared by the test-suite: synthetic reads, probe panels, count files and AnnData objects."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Iterable, Optional, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from giftwrap.pipeline_utils import write_sparse_matrix

FLEX_CONSTANT = "ACGCGGTTAGCACGTA"
FLEX_V2_CONSTANT = "CCCATATAAGAAAACCTGAATACGCGGTT"

LHS_A = "GCTAAAGACAATTACATAACATACA"
RHS_A = "CGTCAGCACGAAACTTGTTGGCCCA"
LHS_B = "GTGTGAATCGCTTAAGGGTTAAGTA"
RHS_B = "AGTGTGATGCATACGCCTTTACTTG"
LHS_C = "GAGGCGCGCCCTCCTGAAGTGCGTG"

CELLS = [
    "GACACTCGCTATGAAT",
    "CTCTGATTTACCCACT",
    "CTGCCAAACTCCAGCG",
    "CGGTCAGTTCCATCAC",
    "CCTAAGTAACCGAATA",
    "ATGCGTTCGCTCTATT",
]
UMIS = [
    "GACTACGACGCG",
    "CTCATTCCCTTG",
    "TCGGAGAGTTAT",
    "GGAACAAGGACG",
    "CTGTCTGAGACT",
    "ACTCTATTTGCC",
    "AATGCGATCCGT",
    "GATTAACTGATA",
]

# Flex v1 probe barcode oligos for BC1 and BC2 (see pipeline_utils._tx_barcode_oligos).
FLEX_BC1 = "ACTTTAGG"
FLEX_BC2 = "AACGGGAA"


def mutate(seq: str, pos: int, base: Optional[str] = None) -> str:
    """Substitute a single base so the result differs from ``seq`` at ``pos``."""
    if base is None:
        base = "A" if seq[pos] != "A" else "C"
    return seq[:pos] + base + seq[pos + 1 :]


def qual(n: int, char: str = "I") -> str:
    return char * n


def flex_r1(cell: str, umi: str, tail: str = "") -> str:
    return cell + umi + tail


def flex_r2(
    lhs: str, gap: str, rhs: str, probe_bc: Optional[str] = None, tail: str = ""
) -> str:
    seq = lhs + gap + rhs + FLEX_CONSTANT
    if probe_bc is not None:
        seq += "NN" + probe_bc
    return seq + tail


def write_fastq(path: Path, records: Iterable[tuple[str, str, str]]) -> Path:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as f:
        for title, seq, q in records:
            f.write(f"@{title}\n{seq}\n+\n{q}\n")
    return path


def write_fastq_pair(
    directory: Path,
    pairs: Sequence[tuple[str, str]],
    name: str = "sample",
    gz: bool = True,
    r1_quals: Optional[Sequence[str]] = None,
    r2_quals: Optional[Sequence[str]] = None,
) -> tuple[Path, Path]:
    """Write paired FASTQs from ``(r1_seq, r2_seq)`` tuples, named like ``sample_S1_R1_001.fastq.gz``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ext = ".fastq.gz" if gz else ".fastq"
    r1 = directory / f"{name}_S1_R1_001{ext}"
    r2 = directory / f"{name}_S1_R2_001{ext}"
    write_fastq(
        r1,
        [
            (f"read{i}", s1, r1_quals[i] if r1_quals else qual(len(s1)))
            for i, (s1, _) in enumerate(pairs)
        ],
    )
    write_fastq(
        r2,
        [
            (f"read{i}", s2, r2_quals[i] if r2_quals else qual(len(s2)))
            for i, (_, s2) in enumerate(pairs)
        ],
    )
    return r1, r2


def write_lines(path: Path, lines: Iterable[str]) -> Path:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as f:
        for line in lines:
            f.write(line + "\n")
    return path


def read_lines(path: Path) -> list[str]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [line.rstrip("\n") for line in f]


def write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> Path:
    return write_lines(
        path, ["\t".join(header)] + ["\t".join(map(str, r)) for r in rows]
    )


def write_whitelist(path: Path, barcodes: Iterable[str], suffix: str = "-1") -> Path:
    """A CellRanger-style ``barcodes.tsv(.gz)``: one barcode per line, no header."""
    return write_lines(path, [f"{bc}{suffix}" for bc in barcodes])


def probe_panel_df(include_control: bool = False) -> pd.DataFrame:
    rows = [
        dict(
            name="GENEA c.1A>T",
            gene="GENEA",
            lhs_probe=LHS_A,
            rhs_probe=RHS_A,
            gap_probe_sequence="TTG",
            original_gap_probe_sequence="TAG",
        ),
        dict(
            name="GENEB c.2C>G",
            gene="GENEB",
            lhs_probe=LHS_B,
            rhs_probe=RHS_B,
            gap_probe_sequence="GCAT",
            original_gap_probe_sequence="GCCT",
        ),
    ]
    if include_control:
        rows.append(
            dict(
                name="GENEC 0bp",
                gene="GENEC",
                lhs_probe=LHS_C,
                rhs_probe=RHS_A,
                gap_probe_sequence="NA",
                original_gap_probe_sequence="NA",
            )
        )
    return pd.DataFrame(rows)


def write_probe_file(path: Path, df: Optional[pd.DataFrame] = None) -> Path:
    path = Path(path)
    df = probe_panel_df() if df is None else df
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, sep="\t")
    return path


def manifest_df(include_control: bool = False) -> pd.DataFrame:
    df = probe_panel_df(include_control).copy()
    df["index"] = range(len(df))
    return df


def _csr(m) -> sp.csr_matrix:
    return sp.csr_matrix(m) if sp.issparse(m) else sp.csr_matrix(np.asarray(m))


def write_counts_h5(
    path: Path,
    X,
    probes: Sequence[tuple[str, str]],
    barcodes: Sequence[str],
    *,
    total_reads=None,
    percent_supporting=None,
    cell_metadata: Optional[dict] = None,
    probe_metadata: Optional[pd.DataFrame] = None,
    cell_index: Optional[Sequence[int]] = None,
    probe_index: Optional[Sequence[int]] = None,
    pcr_layers: Optional[dict[int, object]] = None,
    attrs: Optional[dict] = None,
) -> Path:
    """Write a counts h5 with the exact layout produced by ``giftwrap-collect``."""
    X = _csr(X)
    total_reads = X.copy() if total_reads is None else _csr(total_reads)
    if percent_supporting is None:
        percent_supporting = X.copy().astype(np.float32)
        percent_supporting.data[:] = 1.0
    percent_supporting = _csr(percent_supporting)
    if cell_metadata is None:
        cell_metadata = {"barcode": list(barcodes), "plex_id": [1] * len(barcodes)}
    if probe_metadata is None:
        names = list(dict.fromkeys(p for p, _ in probes))
        probe_metadata = pd.DataFrame(
            {
                "name": names,
                "gene": [n.split(" ")[0] for n in names],
                "lhs_probe": ["L" * 5] * len(names),
                "rhs_probe": ["R" * 5] * len(names),
                "gap_probe_sequence": ["ACG"] * len(names),
                "original_gap_probe_sequence": ["AGG"] * len(names),
                "index": list(range(len(names))),
            }
        )
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcode", data=np.array(list(barcodes), dtype="S"))
        m.create_dataset("probe", data=np.array([list(p) for p in probes], dtype="S"))
        if cell_index is not None:
            m.create_dataset("cell_index", data=np.asarray(cell_index, dtype=np.uint32))
        if probe_index is not None:
            m.create_dataset(
                "probe_index", data=np.asarray(probe_index, dtype=np.uint32)
            )
        write_sparse_matrix(m, "data", X)
        write_sparse_matrix(m, "total_reads", total_reads)
        write_sparse_matrix(m, "percent_supporting", percent_supporting)

        cm = f.create_group("cell_metadata")
        cm.create_dataset(
            "columns", data=np.array(list(cell_metadata.keys()), dtype="S")
        )
        for col, values in cell_metadata.items():
            values = np.asarray(values)
            if not np.issubdtype(values.dtype, np.number):
                values = values.astype("S")
            cm.create_dataset(col, data=values)

        pm = f.create_group("probe_metadata")
        for col in ["name", "gene", "lhs_probe", "rhs_probe", "gap_probe_sequence"]:
            if col in probe_metadata.columns:
                pm.create_dataset(col, data=np.array(probe_metadata[col], dtype="S"))
        pm.create_dataset(
            "original_sequence",
            data=np.array(probe_metadata["original_gap_probe_sequence"], dtype="S"),
        )
        pm.create_dataset(
            "index", data=np.array(probe_metadata["index"], dtype=np.uint32)
        )

        if pcr_layers:
            grp = f.create_group("pcr_thresholded_counts")
            for k, layer in pcr_layers.items():
                write_sparse_matrix(grp, f"pcr{k}", _csr(layer))

        base_attrs = {
            "plex": "1",
            "project": "test",
            "created_date": "2026-01-01 00:00:00",
            "n_cells": len(barcodes),
            "n_probes": len(probe_metadata),
            "n_probe_gapfill_combinations": len(probes),
        }
        if pcr_layers:
            base_attrs["max_pcr_duplicates"] = max(pcr_layers)
        base_attrs.update(attrs or {})
        for k, v in base_attrs.items():
            f.attrs[k] = v
    return Path(path)


def make_gapfill_adata(
    X,
    probes: Sequence[str],
    gapfills: Sequence[str],
    *,
    sparse: bool = True,
    obs_names: Optional[Sequence[str]] = None,
    total_reads=None,
    percent_supporting=None,
    extra_layers: Optional[dict] = None,
    expected: Optional[dict] = None,
    reference: Optional[dict] = None,
    obs: Optional[dict] = None,
) -> ad.AnnData:
    """A gapfill AnnData shaped like ``read_h5_file`` output (var index ``probe|gapfill``)."""
    X = np.asarray(X, dtype=float)
    n_cells = X.shape[0]
    var = pd.DataFrame({"probe": list(probes), "gapfill": list(gapfills)})
    if expected is not None:
        var["expected_gapfill"] = [expected.get(p, "NA") for p in probes]
    if reference is not None:
        var["reference_gapfill"] = [reference.get(p, "NA") for p in probes]
    var.index = [f"{p}|{g}" for p, g in zip(probes, gapfills)]
    obs_df = pd.DataFrame(
        index=(
            list(obs_names)
            if obs_names is not None
            else [f"cell{i}" for i in range(n_cells)]
        )
    )
    for k, v in (obs or {}).items():
        obs_df[k] = v
    conv = (
        (lambda m: sp.csr_matrix(np.asarray(m, dtype=float)))
        if sparse
        else (lambda m: np.asarray(m, dtype=float))
    )
    layers = {
        "total_reads": conv(X * 2 if total_reads is None else total_reads),
        "percent_supporting": conv(
            (X > 0).astype(float) if percent_supporting is None else percent_supporting
        ),
    }
    for k, v in (extra_layers or {}).items():
        layers[k] = conv(v)
    return ad.AnnData(X=conv(X), obs=obs_df, var=var, layers=layers)


def genotype_frame(data: dict, index: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(data, index=list(index), dtype="string").astype("category")


def to_dense(m) -> np.ndarray:
    return m.toarray() if sp.issparse(m) else np.asarray(m)


def write_10x_mtx(
    directory: Path, barcodes: Sequence[str], genes: Sequence[str], matrix=None
) -> Path:
    """A CellRanger ``filtered_feature_bc_matrix`` folder; ``matrix`` is cells x genes (default: ascending ints)."""
    import scipy.io

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if matrix is None:
        matrix = np.arange(1, len(genes) * len(barcodes) + 1).reshape(
            len(barcodes), len(genes)
        )
    buffer = io.BytesIO()
    scipy.io.mmwrite(buffer, sp.coo_matrix(np.asarray(matrix).T))
    with gzip.open(directory / "matrix.mtx.gz", "wb") as f:
        f.write(buffer.getvalue())
    write_lines(
        directory / "features.tsv.gz", [f"ID{g}\t{g}\tGene Expression" for g in genes]
    )
    write_lines(directory / "barcodes.tsv.gz", list(barcodes))
    return directory
