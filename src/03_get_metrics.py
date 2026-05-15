import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "processed", "02_pns2019_selected_columns.csv")
df = pd.read_csv(csv_path)

def classificar_tipo(series):
    """
    Classifica coluna como:
    - categorico
    - categorico_ordinal
    - numerico_discreto
    - numerico_continuo
    """

    # remove NaN para análise
    s = series.dropna()

    if len(s) == 0:
        return "categorico"

    n_distinct = s.nunique()

    # regra principal:
    # poucos valores distintos => categórico
    if n_distinct <= 10:

        # inteiro pequeno geralmente ordinal/codificado
        if pd.api.types.is_numeric_dtype(s):
            return "categorico_ordinal"

        return "categorico"

    # floats
    if pd.api.types.is_float_dtype(s):

        # float com poucos valores únicos
        # ex: 0.0, 1.0, 2.0
        if n_distinct <= 20:
            return "categorico_ordinal"

        return "numerico_continuo"

    # inteiros
    if pd.api.types.is_integer_dtype(s):
        return "numerico_discreto"

    # strings/objects
    return "categorico"

def metricas_coluna(series):
    total = len(series)
    ausentes = series.isna().sum()
    preenchidos = total - ausentes
    n_distinct = series.nunique(dropna=True)
    tipo = classificar_tipo(series)

    info = {
        "tipo": tipo,
        "total_registros": total,
        "preenchidos": int(preenchidos),
        "ausentes": int(ausentes),
        "pct_ausentes": round(ausentes / total * 100, 2),
        "n_distinct": int(n_distinct),
    }

    if tipo in ("numerico_continuo", "numerico_discreto"):
        info.update({
            "media":   round(series.mean(), 4)   if preenchidos > 0 else None,
            "mediana": round(series.median(), 4) if preenchidos > 0 else None,
            "desvio_padrao": round(series.std(), 4) if preenchidos > 0 else None,
            "minimo":  round(series.min(), 4)    if preenchidos > 0 else None,
            "maximo":  round(series.max(), 4)    if preenchidos > 0 else None,
            "q25":     round(series.quantile(0.25), 4) if preenchidos > 0 else None,
            "q75":     round(series.quantile(0.75), 4) if preenchidos > 0 else None,
            "skewness": round(series.skew(), 4)  if preenchidos > 0 else None,
            "kurtosis": round(series.kurt(), 4)  if preenchidos > 0 else None,
        })
    elif tipo == "categorico_ordinal":
        info.update({
            "media":   round(series.mean(), 4)   if preenchidos > 0 else None,
            "mediana": round(series.median(), 4) if preenchidos > 0 else None,
            "desvio_padrao": round(series.std(), 4) if preenchidos > 0 else None,
            "moda":    series.mode().tolist()[0]  if preenchidos > 0 else None,
            "distribuicao": series.value_counts(dropna=True).to_dict(),
        })
    else:  # categorico puro (string/object)
        moda = series.mode().tolist()
        info.update({
            "moda": moda[0] if moda else None,
            "distribuicao": series.value_counts(dropna=True).head(20).to_dict(),
        })

    return info

# Gera métricas para todas as colunas
resultado = {}
for col in df.columns:
    resultado[col] = metricas_coluna(df[col])

# Exporta JSON
output_path = os.path.join(BASE_DIR, "data", "processed", "03_pns2019_column_metrics.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2, default=str)

print(f"Métricas exportadas para {output_path}")
print(f"Total de atributos analisados: {len(resultado)}")

# Preview resumido no terminal
print("\n--- Resumo ---")
for col, m in resultado.items():
    print(f"{col:15s} | {m['tipo']:22s} | ausentes: {m['pct_ausentes']:5.1f}% | distinct: {m['n_distinct']}")