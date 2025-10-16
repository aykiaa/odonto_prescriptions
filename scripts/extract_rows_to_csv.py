#!/usr/bin/env python3
import argparse
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default="/scratch/arturxavier/odonto_prescricoes/prescricoes_all.parquet",
        help="Caminho do Parquet consolidado",
    )
    parser.add_argument(
        "--out",
        default="/scratch/arturxavier/odonto_prescricoes/prescricoes_sample_50.csv",
        help="Caminho do CSV de saída",
    )
    parser.add_argument("--n", type=int, default=50, help="Número de linhas a extrair")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {parquet_path}")

    lf = pl.scan_parquet(str(parquet_path))
    df = lf.head(args.n).collect()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)
    print(f"CSV salvo em: {out_path} ({len(df)} linhas)")


if __name__ == "__main__":
    main()


