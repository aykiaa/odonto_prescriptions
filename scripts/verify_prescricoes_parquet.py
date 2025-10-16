#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import polars as pl


def normalize_key_expr(s: pl.Expr) -> pl.Expr:
    return (
        s.cast(pl.Utf8)
        .str.to_uppercase()
        .str.strip_chars()
        .str.replace(r"\s+", " ", literal=False)
    )


def read_dict_descriptor_cols(dict_path: str) -> List[str]:
    df = pl.read_csv(
        dict_path,
        separator=";",
        encoding="latin1",
        infer_schema_length=0,
        ignore_errors=True,
    )
    key_cols = [c for c in ["PRINCIPIO_ATIVO", "PRINCIPIO_ATIVO_1", "PRINCIPIO_ATIVO_2"] if c in df.columns]
    descriptor_cols = [c for c in df.columns if c not in key_cols and not str(c).startswith("norm_")]
    return descriptor_cols


def verify_parquet(parquet_path: str, dict_path: str, sample_rows: int = 5) -> None:
    path = Path(parquet_path)
    if not path.exists():
        raise SystemExit(f"Arquivo não encontrado: {parquet_path}")

    lf = pl.scan_parquet(parquet_path)
    schema = lf.schema
    cols = list(schema.keys())

    print("Resumo do arquivo Parquet:")
    total_rows = lf.select(pl.len()).collect().item()
    print(f"- linhas: {total_rows}")
    print(f"- colunas: {len(cols)}")
    print(f"- campos: {cols}")

    # Amostra
    try:
        sample_df = lf.fetch(sample_rows)
        print(f"Amostra ({len(sample_df)} linhas):")
        print(sample_df)
    except Exception as e:
        print(f"Aviso: falha ao obter amostra: {e}")

    # Checagens de colunas essenciais
    required = ["ID", "ano", "PRINCIPIO_ATIVO"]
    missing_req = [c for c in required if c not in cols]
    if missing_req:
        print(f"ERRO: colunas obrigatórias ausentes: {missing_req}")
    else:
        print("Colunas obrigatórias presentes: OK")

    # Unicidade do ID
    if "ID" in cols:
        dup = (
            lf.group_by("ID").len().filter(pl.col("len") > 1).select(pl.len().alias("n_dups")).collect().item()
        )
        print(f"IDs duplicados: {dup}")

        # Formato do ID e consistência com ano (prefixo AAAA-)
        if "ano" in cols:
            # Checar que prefixo do ID é o ano + '-'
            mismatches = (
                lf.select(
                    (
                        (pl.col("ID").str.slice(0, 5) != (pl.col("ano") + pl.lit("-")))
                    ).sum().alias("n_mismatch")
                ).collect().item()
            )
            print(f"IDs com prefixo divergente de 'ano-': {mismatches}")

    # Distribuição por ano
    if "ano" in cols:
        by_year = lf.group_by("ano").len().sort("ano").collect()
        print("Linhas por ano:")
        print(by_year)

    # Cobertura do dicionário
    descriptor_cols = read_dict_descriptor_cols(dict_path)
    dict_missing = [c for c in descriptor_cols if c not in cols]
    if dict_missing:
        print(f"Aviso: colunas do dicionário ausentes no Parquet: {dict_missing}")

    if "key_norm" in cols:
        matched = lf.select((pl.col("key_norm").is_not_null()).sum().alias("mapeados")).collect().item()
        print(f"Registros mapeados no dicionário: {matched} de {total_rows} ({matched/total_rows:.2%})")
        # Principais não mapeados
        if "PRINCIPIO_ATIVO" in cols:
            top_unmapped = (
                lf.filter(pl.col("key_norm").is_null())
                .group_by("ano", "PRINCIPIO_ATIVO")
                .len()
                .sort("len", descending=True)
                .limit(10)
                .collect()
            )
            print("Top 10 PRINCIPIO_ATIVO não mapeados por contagem:")
            print(top_unmapped)
    else:
        print("Aviso: coluna 'key_norm' não encontrada; não é possível avaliar cobertura do dicionário.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default="/scratch/arturxavier/odonto_prescricoes/prescricoes_all.parquet",
        help="Caminho do Parquet consolidado",
    )
    parser.add_argument(
        "--dict",
        default="/scratch/arturxavier/odonto_prescricoes/analise_geral_medicamentos.csv",
        help="Caminho do dicionário de medicamentos (CSV ; latin1)",
    )
    parser.add_argument("--sample", type=int, default=5, help="Tamanho da amostra a imprimir")
    args = parser.parse_args()

    verify_parquet(args.parquet, args.dict, sample_rows=args.sample)


if __name__ == "__main__":
    main()


