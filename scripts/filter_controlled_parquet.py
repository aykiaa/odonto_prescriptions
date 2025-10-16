#!/usr/bin/env python3
import argparse
from pathlib import Path

import polars as pl


def normalize_text_expr(s: pl.Expr) -> pl.Expr:
    return (
        s.cast(pl.Utf8)
        .str.to_lowercase()
        .str.strip_chars()
        .str.replace(r"\s+", " ", literal=False)
    )


def build_filter_expr(cols: list[str]) -> pl.Expr:
    """Return an expression selecting A/S/H by dictionary flag or any class containing 'opio'."""
    cond_ahs = pl.lit(False)
    flag_col = "Ansiolítico/Sedativo/Hipnótico"
    if flag_col in cols:
        cond_ahs = normalize_text_expr(pl.col(flag_col)).eq("sim")

    cond_opio = pl.lit(False)
    for c in ["Classe_1", "Classe_2", "Classe_3", "Classe_4"]:
        if c in cols:
            cond_opio = cond_opio | normalize_text_expr(pl.col(c)).str.contains("opio")

    return cond_ahs | cond_opio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default="/scratch/arturxavier/odonto_prescricoes/prescricoes_all.parquet",
        help="Caminho do Parquet consolidado de entrada",
    )
    parser.add_argument(
        "--out",
        default="/scratch/arturxavier/odonto_prescricoes/prescricoes_ansio_hipno_sed_opioides.parquet",
        help="Caminho do Parquet filtrado de saída",
    )
    args = parser.parse_args()

    in_path = Path(args.parquet)
    if not in_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {in_path}")

    lf = pl.scan_parquet(str(in_path))
    cols = lf.columns

    filt = build_filter_expr(cols)
    out_lf = lf.filter(filt)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_lf.sink_parquet(str(out_path), compression="snappy")

    # Print basic stats
    total = lf.select(pl.len()).collect().item()
    kept = out_lf.select(pl.len()).collect().item()
    print(f"Filtrado: {kept} de {total} linhas -> {kept/total:.2%}")
    print(f"Parquet salvo em: {out_path}")


if __name__ == "__main__":
    main()


