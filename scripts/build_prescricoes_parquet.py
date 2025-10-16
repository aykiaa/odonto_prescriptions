#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import polars as pl
import chardet


def detect_sep(first_line: str) -> str:
    return ";" if first_line.count(";") >= first_line.count(",") else ","


def normalize_key_expr(s: pl.Expr) -> pl.Expr:
    # Uppercase, trim, collapse internal whitespace
    return (
        s.cast(pl.Utf8)
        .str.to_uppercase()
        .str.strip_chars()
        .str.replace(r"\s+", " ", literal=False)
    )


def sanitize_df_columns(df: pl.DataFrame) -> pl.DataFrame:
    # Drop empty-named columns (can appear from trailing separators)
    drop_cols = [c for c in df.columns if c is None or str(c).strip() == ""]
    if drop_cols:
        df = df.drop(drop_cols)
    return df


def build_dict_table(dict_path: str) -> tuple[pl.LazyFrame, List[str]]:
    # Read with proper encoding and semicolon separator
    df = pl.read_csv(
        dict_path,
        separator=";",
        encoding="latin1",
        infer_schema_length=0,
        ignore_errors=True,
    )
    df = sanitize_df_columns(df)

    # Expected alias columns
    key_cols = [c for c in ["PRINCIPIO_ATIVO", "PRINCIPIO_ATIVO_1", "PRINCIPIO_ATIVO_2"] if c in df.columns]
    if not key_cols:
        raise SystemExit("Dicionário sem colunas de chave esperadas: PRINCIPIO_ATIVO(_1|_2)")

    # Normalize aliases
    for c in key_cols:
        df = df.with_columns(normalize_key_expr(pl.col(c)).alias(f"norm_{c}"))

    alias_norm_cols = [f"norm_{c}" for c in key_cols]

    # Descriptor columns are everything except the alias (raw and normalized) columns
    descriptor_cols = [
        c for c in df.columns if c not in key_cols and not str(c).startswith("norm_")
    ]

    # Unpivot to long one row per alias (replacement for deprecated melt)
    long = df.unpivot(
        index=descriptor_cols,
        on=alias_norm_cols,
        variable_name="alias_col",
        value_name="key_norm",
    )
    long = long.filter(pl.col("key_norm").is_not_null() & (pl.col("key_norm") != ""))
    long = long.unique(subset=["key_norm"])  # keep first occurrence per normalized key
    # Drop internal helper column from melt
    if "alias_col" in long.columns:
        long = long.drop("alias_col")

    # Return LazyFrame and descriptor column names to detect match later if needed
    return long.lazy(), descriptor_cols


def read_header(path: str) -> Tuple[List[str], str]:
    with open(path, "rb") as f:
        head = f.read(8192)
    enc = chardet.detect(head).get("encoding") or "utf-8"
    try:
        first_line = head.decode(enc, errors="ignore").splitlines()[0]
    except Exception:
        first_line = ""
    sep = detect_sep(first_line)
    # Try to read zero rows just to get schema
    cols: List[str]
    try:
        df0 = pl.read_csv(path, separator=sep, n_rows=0, ignore_errors=True)
        cols = df0.columns
    except Exception:
        cols = [c.strip() for c in first_line.split(sep)] if first_line else []
    return cols, sep


def year_from_filename(path: str) -> str:
    m = re.search(r"(20\d{2})", os.path.basename(path))
    if not m:
        raise ValueError(f"Não foi possível extrair ano de {path}")
    return m.group(1)


def build_lazy_for_csv(
    path: str,
    superset_cols: List[str],
    sep: str,
    dict_map: pl.LazyFrame,
) -> pl.LazyFrame:
    ano = year_from_filename(path)
    lf = pl.scan_csv(path, separator=sep, ignore_errors=True)

    # Ensure every superset column exists and is Utf8 for schema stability
    selected = []
    lf_cols = set(lf.collect_schema().names())
    for c in superset_cols:
        if c in lf_cols:
            selected.append(pl.col(c).cast(pl.Utf8, strict=False).alias(c))
        else:
            selected.append(pl.lit(None).cast(pl.Utf8).alias(c))
    lf = lf.select(selected)

    # Add ano, row_number (1-based) and ID
    lf = lf.with_columns(pl.lit(ano).alias("ano"))
    lf = lf.with_row_index(name="row_number", offset=1)
    lf = lf.with_columns(
        (pl.col("ano") + pl.lit("-") + pl.col("row_number").cast(pl.Utf8).str.zfill(8)).alias("ID")
    )

    # Normalize join key and left join dict mapping
    if "PRINCIPIO_ATIVO" in superset_cols:
        lf = lf.with_columns(normalize_key_expr(pl.col("PRINCIPIO_ATIVO")).alias("key_norm"))
        lf = lf.join(dict_map, on="key_norm", how="left")
    else:
        lf = lf.with_columns(pl.lit(None).alias("key_norm"))

    return lf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="/scratch/arturxavier/odonto_prescricoes/grouped_year_filtered",
        help="Diretório com grouped_YYYY.csv",
    )
    parser.add_argument(
        "--dict",
        default="/scratch/arturxavier/odonto_prescricoes/analise_geral_medicamentos.csv",
        help="Caminho do dicionário de medicamentos (CSV ; latin1)",
    )
    parser.add_argument(
        "--output",
        default="/scratch/arturxavier/odonto_prescricoes/prescricoes_all.parquet",
        help="Caminho de saída Parquet",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in input_dir.glob("grouped_*.csv"))
    if not files:
        raise SystemExit("Nenhum CSV encontrado em --input-dir")

    # Build dictionary mapping
    dict_map, _descriptor_cols = build_dict_table(args.dict)

    # Discover superset of columns and per-file separator
    file_meta: list[tuple[str, list[str], str]] = []
    col_set: set[str] = set()
    for fp in files:
        cols, sep = read_header(fp)
        if not cols:
            raise SystemExit(f"Cabeçalho vazio em {fp}")
        col_set.update(cols)
        file_meta.append((fp, cols, sep))

    superset_cols = sorted(col_set)

    # Build lazy frames per file
    lazy_frames: list[pl.LazyFrame] = []
    for fp, _cols, sep in file_meta:
        lf = build_lazy_for_csv(fp, superset_cols, sep, dict_map)
        lazy_frames.append(lf)

    # Union
    all_lf = pl.concat(lazy_frames, how="vertical_relaxed")

    # Drop columns deemed duplicate/technical prior to writing
    cols_to_drop = [
        "ANO_VENDA",  # duplicate of 'ano'
        "row_number",  # technical (used only for ID)
        "key_norm",  # technical (join helper)
        "_duplicated_0",  # artifact from joins
    ]
    drop_existing = [c for c in cols_to_drop if c in all_lf.collect_schema().names()]
    if drop_existing:
        all_lf = all_lf.drop(drop_existing)

    # Reorder columns so that ID, ano and DESCRICAO_APRESENTACAO come first
    schema_cols = all_lf.collect_schema().names()
    first_order = [c for c in ["ID", "ano", "DESCRICAO_APRESENTACAO"] if c in schema_cols]
    if first_order:
        all_lf = all_lf.select([*(pl.col(c) for c in first_order), pl.all().exclude(first_order)])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_lf.sink_parquet(str(out_path), compression="snappy")
    print(f"Parquet escrito em: {out_path}")

    # Optional: unmatched report
    try:
        # Compute per-year counts of unmatched PRINCIPIO_ATIVO
        if "PRINCIPIO_ATIVO" in superset_cols:
            unmatched = (
                all_lf
                .with_columns([
                    pl.col("key_norm").is_null().alias("_unmatched"),
                ])
                .group_by(["ano", "_unmatched"]).len()
                .filter(pl.col("_unmatched") == True)
                .select([pl.col("ano"), pl.col("len").alias("nao_mapeados")])
            )
            out_csv = out_path.with_name("dict_unmatched_por_ano.csv")
            unmatched.sink_csv(str(out_csv))
            print(f"Relatório de não mapeados: {out_csv}")
    except Exception as e:
        # Non-fatal
        print(f"Aviso: falha ao gerar relatório de não mapeados: {e}")


if __name__ == "__main__":
    main()


