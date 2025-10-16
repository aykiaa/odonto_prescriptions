#!/usr/bin/env python3
import argparse
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import polars as pl


TECHNICAL_COL_PREFIXES = ("_duplicated_",)
TECHNICAL_COLS = {"row_number", "key_norm"}
YEAR_COL_CANONICAL = "ano"
YEAR_COL_ALTERNATIVE = "ANO_VENDA"


def normalize_text_expr(s: pl.Expr) -> pl.Expr:
    return (
        s.cast(pl.Utf8)
        .str.to_lowercase()
        .str.strip_chars()
        .str.replace(r"\s+", " ", literal=False)
    )


def compute_basic_stats(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for c in df.columns:
        exprs.extend(
            [
                pl.col(c).is_null().sum().alias(f"{c}__nulls"),
                pl.col(c).n_unique().alias(f"{c}__n_unique"),
            ]
        )
    return df.select(exprs)


def pairwise_mismatch_counts(df: pl.DataFrame, cols: List[str]) -> list[tuple[str, str, int]]:
    results: list[tuple[str, str, int]] = []
    # Compare on normalized string view with nulls filled to sentinel to consider null==null
    sentinel = "\u2400NULL\u2400"
    n = len(df)
    for a, b in combinations(cols, 2):
        mismatch = (
            (
                pl.col(a)
                .cast(pl.Utf8)
                .fill_null(sentinel)
                .str.strip_chars()
                .str.replace(r"\s+", " ", literal=False)
                != pl.col(b)
                .cast(pl.Utf8)
                .fill_null(sentinel)
                .str.strip_chars()
                .str.replace(r"\s+", " ", literal=False)
            )
            .sum()
            .alias("mismatch")
        )
        m = df.select(mismatch).item()
        if m == 0:
            results.append((a, b, m))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default="/scratch/arturxavier/odonto_prescricoes/prescricoes_all.parquet",
        help="Caminho do Parquet consolidado",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200_000,
        help="N de linhas para amostra (para análises intensivas)",
    )
    args = parser.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        raise SystemExit(f"Arquivo não encontrado: {path}")

    lf = pl.scan_parquet(str(path))
    total_rows = lf.select(pl.len()).collect().item()
    cols = lf.columns

    print(f"Arquivo: {path}")
    print(f"Linhas totais: {total_rows}")
    print(f"Colunas ({len(cols)}): {cols}")

    # Identify technical columns and duplicated-prefixed columns
    technical = [c for c in cols if c in TECHNICAL_COLS or any(c.startswith(p) for p in TECHNICAL_COL_PREFIXES)]
    if technical:
        print(f"Colunas técnicas/auxiliares (candidatas a remoção): {technical}")

    # Exact duplicate content checks for specific known pairs (year)
    if YEAR_COL_CANONICAL in cols and YEAR_COL_ALTERNATIVE in cols:
        mismatches = (
            lf.select(
                (
                    (
                        pl.col(YEAR_COL_CANONICAL).cast(pl.Utf8).fill_null("NULL")
                        != pl.col(YEAR_COL_ALTERNATIVE).cast(pl.Utf8).fill_null("NULL")
                    )
                ).sum().alias("mismatches")
            )
            .collect()
            .item()
        )
        if mismatches == 0:
            print(f"Colunas duplicadas por conteúdo: '{YEAR_COL_ALTERNATIVE}' == '{YEAR_COL_CANONICAL}' em todas as linhas")
        else:
            print(f"Divergências entre '{YEAR_COL_CANONICAL}' e '{YEAR_COL_ALTERNATIVE}': {mismatches}")

    # Sample for broader stats and pairwise checks
    sample_df = lf.head(args.limit).collect()
    print(f"Amostra usada para estatísticas: {len(sample_df)} linhas")

    # Basic stats per column
    stats = compute_basic_stats(sample_df)
    # Summarize
    rec_drop: list[str] = []
    for c in sample_df.columns:
        nulls = stats.select(pl.col(f"{c}__nulls")).item()
        nunique = stats.select(pl.col(f"{c}__n_unique")).item()
        null_ratio = nulls / len(sample_df)
        if c in technical:
            rec_drop.append(c)
        elif nunique <= 1:
            rec_drop.append(c)
        elif null_ratio > 0.99:
            rec_drop.append(c)

    # Pairwise equal columns on sample
    equal_pairs = pairwise_mismatch_counts(sample_df, sample_df.columns)
    if equal_pairs:
        print("Pares de colunas idênticas (na amostra):")
        for a, b, _ in equal_pairs:
            print(f"- {a} == {b}")
            # Prefer keep canonical 'ano' instead of 'ANO_VENDA'
            if {a, b} == {YEAR_COL_CANONICAL, YEAR_COL_ALTERNATIVE}:
                rec = YEAR_COL_ALTERNATIVE if YEAR_COL_CANONICAL in cols else YEAR_COL_CANONICAL
                if rec not in rec_drop:
                    rec_drop.append(rec)

    print("Sugestões de remoção (heurísticas):")
    if rec_drop:
        print(sorted(set(rec_drop)))
    else:
        print("Nenhuma coluna candidata encontrada com as heurísticas atuais.")


if __name__ == "__main__":
    main()


