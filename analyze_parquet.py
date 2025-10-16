import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FILE_PATH = 'prescricoes_all.parquet'

pf = pq.ParquetFile(FILE_PATH)
columns = pf.schema.names

null_counts = {col: 0 for col in columns}
empty_counts = {col: 0 for col in columns}

categorical_counters = {
    'SEXO': Counter(),
    'TIPO_RECEITUARIO': Counter(),
    'UNIDADE_IDADE': Counter(),
    'UNIDADE_MEDIDA': Counter(),
    'CONSELHO_PRESCRITOR': Counter(),
    'UF_CONSELHO_PRESCRITOR': Counter(),
    'UF_VENDA': Counter(),
    'Ansiolítico/Sedativo/Hipnótico': Counter(),
}

numeric_specs = {
    'ano': {'min': math.inf, 'max': -math.inf, 'invalid': 0, 'non_integer': 0, 'count': 0},
    'MES_VENDA': {'min': math.inf, 'max': -math.inf, 'invalid': 0, 'non_integer': 0, 'count': 0},
    'IDADE': {'min': math.inf, 'max': -math.inf, 'invalid': 0, 'non_integer': 0, 'count': 0},
    'QTD_VENDIDA': {'min': math.inf, 'max': -math.inf, 'invalid': 0, 'non_integer': 0, 'count': 0, 'zero': 0, 'negative': 0},
}

range_limits = {
    'MES_VENDA': (1, 12),
    'IDADE': (0, 120),
}

range_issues = {col: 0 for col in range_limits}

pattern_issues = {
    'ID': 0,
    'CID10': 0,
}

id_pattern = re.compile(r'^\d{4}-\d{8}$')
cid10_pattern = re.compile(r'^[A-Z]\d{2}[A-Z0-9]?(?:\.\d{1,4})?$')

rows_processed = 0
for rg_index in range(pf.num_row_groups):
    table = pf.read_row_group(rg_index)
    df = table.to_pandas()
    rows_processed += len(df)

    for col in columns:
        series = df[col]
        null_counts[col] += series.isna().sum()
        if series.notna().any():
            empty_counts[col] += series.dropna().astype(str).str.strip().eq('').sum()

    # categorical counters
    for col, counter in categorical_counters.items():
        if col in df.columns:
            values = df[col].dropna().astype(str).str.strip()
            counter.update(values)

    # numeric-like columns
    for col, spec in numeric_specs.items():
        if col not in df.columns:
            continue
        col_series = df[col]
        if col_series.notna().any():
            cleaned = col_series.dropna().astype(str).str.replace(',', '.', regex=False)
            numeric = pd.to_numeric(cleaned, errors='coerce')
            spec['invalid'] += numeric.isna().sum()
            valid = numeric.dropna()
            spec['count'] += len(valid)
            if not valid.empty:
                min_val = valid.min()
                max_val = valid.max()
                if min_val < spec['min']:
                    spec['min'] = min_val
                if max_val > spec['max']:
                    spec['max'] = max_val
                if (valid % 1 != 0).any():
                    spec['non_integer'] += (valid % 1 != 0).sum()
                if col in range_limits:
                    lower, upper = range_limits[col]
                    range_issues[col] += ((valid < lower) | (valid > upper)).sum()
                if col == 'QTD_VENDIDA':
                    spec['zero'] += (valid == 0).sum()
                    spec['negative'] += (valid < 0).sum()

    # pattern checks
    if 'ID' in df.columns:
        ids = df['ID'].dropna().astype(str)
        pattern_issues['ID'] += (~ids.str.match(id_pattern)).sum()
    if 'CID10' in df.columns:
        cid_values = df['CID10'].dropna().astype(str).str.upper()
        pattern_issues['CID10'] += (~cid_values.str.match(cid10_pattern)).sum()

print(f'Total rows observed: {rows_processed}')
print('\nMissing values per column:')
for col in columns:
    missing = null_counts[col]
    empty = empty_counts[col]
    print(f'  {col}: nulls={missing:,} ({missing / rows_processed:.2%}), empty={empty:,} ({empty / rows_processed:.2%})')

print('\nNumeric-like column checks:')
for col, spec in numeric_specs.items():
    print(f'  {col}:')
    print(f'    valid count: {spec["count"]:,}')
    print(f'    invalid parse: {spec["invalid"]:,}')
    if spec['count']:
        print(f'    min={spec["min"]}, max={spec["max"]}')
    print(f'    non integer values: {spec["non_integer"]:,}')
    if col in range_issues:
        print(f'    out of expected range: {range_issues[col]:,}')
    if col == 'QTD_VENDIDA':
        print(f'    zero qty: {spec["zero"]:,}')
        print(f'    negative qty: {spec["negative"]:,}')

print('\nPattern issues:')
for key, value in pattern_issues.items():
    print(f'  {key}: {value:,}')

print('\nCategorical value counts (top 10):')
for col, counter in categorical_counters.items():
    print(f'  {col}:')
    for value, count in counter.most_common(10):
        print(f'    {value}: {count:,}')

# Additional derived insights
if numeric_specs['ano']['count']:
    years = []
    for col_value, count in categorical_counters.get('ano', Counter()).items():
        years.append((col_value, count))

