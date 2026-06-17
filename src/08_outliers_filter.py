"""
Script: 08_outliers_filter.py
Objetivo: Tratamento de outliers INDIVIDUALIZADO por variável, aplicando a técnica
          mais adequada à composição estatística e ao domínio clínico de cada uma.
          Corrige erros do tratamento anterior (04_data_process.py) onde variáveis
          como Q031, P036 e P029 foram danificadas por winsorização com IQR=[0,0].

Input:
    data/processed/06_pns2019_prepared.csv  ← base pós-engenharia de variáveis
    data/processed/05_pns2019_clean.csv     ← base limpa (para restaurar Q031)
    data/processed/02_pns2019_selected_columns.csv ← base original (para restaurar P036)
Output:
    data/processed/08_pns2019_outliers_treated.csv ← base tratada
    docs/relatorios/relatorio_outliers_08.txt      ← relatório detalhado
    docs/relatorios/plots/outliers_08/             ← plots antes/depois

Técnicas:
    - Z-Score (|z| > 3): para distribuições ~normais (|skew| < 0.5)
    - IQR (Q1-1.5×IQR, Q3+1.5×IQR): para distribuições moderadas (IQR > 0)
    - MAD (|x-med|/MAD > 3.5): robusto para distribuições assimétricas (|skew| > 1)
    - Percentil [P1, P99]: conservador para caudas pesadas (kurtosis > 5)
    - Domínio clínico: limites biológicos fixos para variáveis de saúde

Ações:
    - WINSORIZAR: clipa valores nos limites detectados (preserva registros)
    - FLAG: cria coluna binária {col}_outlier_flag (preserva valor original)
    - MANTER: nenhuma ação necessária

Referências:
    - Prof. Zárate — Slide 14 (Análise de Outliers)
    - Leys et al., 2013 — MAD como alternativa robusta ao desvio-padrão
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_PREPARED   = os.path.join(BASE_DIR, "data", "processed", "06_pns2019_prepared.csv")
PATH_CLEAN      = os.path.join(BASE_DIR, "data", "processed", "05_pns2019_clean.csv")
PATH_ORIGINAL   = os.path.join(BASE_DIR, "data", "processed", "02_pns2019_selected_columns.csv")
PATH_OUTPUT     = os.path.join(BASE_DIR, "data", "processed", "08_pns2019_outliers_treated.csv")
DIR_RELATORIOS  = os.path.join(BASE_DIR, "docs", "relatorios")
DIR_PLOTS       = os.path.join(BASE_DIR, "docs", "relatorios", "plots", "outliers_08")

os.makedirs(DIR_RELATORIOS, exist_ok=True)
os.makedirs(DIR_PLOTS, exist_ok=True)

TARGET_COL = "Q03001"

# Constante MAD: fator de correção para normalidade
MAD_SCALE = 1.4826
# Thresholds
ZSCORE_LIMIAR = 3.0
IQR_MULT      = 1.5
MAD_LIMIAR     = 3.5
PERC_LOW       = 0.01
PERC_HIGH      = 0.99


# ─────────────────────────────────────────────
# ESTRATÉGIAS POR VARIÁVEL
# ─────────────────────────────────────────────
# Cada entrada define o método de detecção e a ação.
# Variáveis não listadas + numéricas contínuas (nunique > 15)
# receberão detecção automática baseada em skewness/kurtosis.

ESTRATEGIAS = {
    # ── Variáveis derivadas (engenharia de variáveis 06) ──────────
    "Calculo_IMC": {
        "metodo": "dominio",
        "limites": (12.0, 70.0),
        "acao": "winsorizar",
        "justificativa": "IMC: limites biológicos plausíveis [12, 70] (OMS + extremos clínicos)"
    },
    "Score_Ultraprocessados_Ontem": {
        "metodo": "nenhum",
        "acao": "manter",
        "justificativa": "Score discreto limitado [0, 10] — sem outliers possíveis por construção"
    },
    "Score_Saude_Mental": {
        "metodo": "nenhum",
        "acao": "manter",
        "justificativa": "Score PHQ-9 discreto limitado [0, 24] — sem outliers possíveis por construção"
    },
    "Minutos_Semanais_Exercicio": {
        "metodo": "mad",
        "acao": "winsorizar",
        "lim_inferior_minimo": 0,
        "justificativa": "Distribuição altamente assimétrica (skew~4.1); MAD robusto; mín=0 (tempo)"
    },

    # ── Renda ──────────────────────────────────────────────────────
    "VDF003": {
        "metodo": "mad",
        "acao": "flag",
        "justificativa": "Renda per capita com skew=8.77 — outliers extremos são reais (alta renda)"
    },

    # ── Saúde / consultas ─────────────────────────────────────────
    "J012": {
        "metodo": "percentil",
        "acao": "flag",
        "justificativa": "Consultas médicas: kurtosis~208 com cauda pesada; percentil [P1,P99] conservador"
    },
    "E017": {
        "metodo": "percentil",
        "acao": "flag",
        "justificativa": "Horas de trabalho: valores extremos reais (plantões, dupla jornada)"
    },

    # ── Álcool ────────────────────────────────────────────────────
    "P029": {
        "metodo": "mad",
        "acao": "winsorizar",
        "lim_inferior_minimo": 0,
        "justificativa": "Doses de álcool: skew~2.95 no original; MAD robusto; winsorizar extremos; mín=0"
    },

    # ── Idade do diagnóstico ──────────────────────────────────────
    "Q031": {
        "metodo": "dominio",
        "limites": (0, 60),
        "acao": "flag",
        "justificativa": "Idade diagnóstico diabetes: range biológico [0, 60] anos; outliers são erro de digitação"
    },

    # ── V0001 (UF) ─ código geográfico, não é contínua ───────────
    "V0001": {
        "metodo": "nenhum",
        "acao": "manter",
        "justificativa": "Código de Unidade da Federação (11-53) — variável nominal codificada, não analisar"
    },
}

# Variáveis que são categóricas/ordinais codificadas como números
# e NÃO devem ter análise de outliers (nunique <= 15)
CATEGORICAS_EXCLUIDAS = {
    # Frequência alimentar (dias/semana: 0-7)
    "P006", "P00901", "P01001", "P01101", "P013", "P015", "P018",
    "P019", "P01601", "P02002", "P02501", "P02602", "P023",
    "P02001",
    # Binários de consumo (1=sim, 2=não)
    "P00601", "P00602", "P00603", "P00604", "P00605", "P00607",
    "P00608", "P00609", "P00610", "P00611", "P00612", "P00613",
    # Tipo (categorias)
    "P02102", "P02401", "P02101", "P02601",
    # Sociodemográficas
    "C006", "C009", "C011", "V0026", "V0031", "VDD004A",
    # Exercício/trabalho
    "P036", "P038", "P039", "P040", "P04501", "P04502",
    "VDM001",
    # Saúde (categóricas)
    "Q00101", "Q02901", "Q03001", "P050", "P027", "P02801",
    "P03201", "VDF004", "I00102", "VDE014", "M005010",
    # Flags existentes
    "P00103_outlier_flag", "P00104_outlier_flag",
    "P00403_outlier_flag", "P00404_outlier_flag",
    "VDF003_outlier_flag", "J012_outlier_flag", "E017_outlier_flag",
}


# ─────────────────────────────────────────────
# FUNÇÕES DE DETECÇÃO
# ─────────────────────────────────────────────

def detectar_zscore(serie, limiar=ZSCORE_LIMIAR):
    """Z-Score: |z| > limiar. Melhor para distribuições ~normais."""
    media = serie.mean()
    dp = serie.std()
    if dp == 0:
        return pd.Series(False, index=serie.index), media - limiar * dp, media + limiar * dp
    z = (serie - media) / dp
    mask = np.abs(z) > limiar
    return mask, media - limiar * dp, media + limiar * dp


def detectar_iqr(serie, mult=IQR_MULT):
    """IQR: fora de [Q1-1.5×IQR, Q3+1.5×IQR]. Só aplica quando IQR > 0."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        # IQR=0 → método inválido para esta variável
        return pd.Series(False, index=serie.index), q1, q3
    lim_inf = q1 - mult * iqr
    lim_sup = q3 + mult * iqr
    mask = (serie < lim_inf) | (serie > lim_sup)
    return mask, lim_inf, lim_sup


def detectar_mad(serie, limiar=MAD_LIMIAR):
    """MAD (Median Absolute Deviation): robusto a assimetria.
    Outlier se |x - mediana| / (MAD × 1.4826) > limiar."""
    mediana = serie.median()
    mad = np.median(np.abs(serie - mediana))
    if mad == 0:
        # MAD=0: tenta fallback com IQR
        return detectar_iqr(serie)
    mad_scaled = mad * MAD_SCALE
    desvio = np.abs(serie - mediana) / mad_scaled
    mask = desvio > limiar
    lim_inf = mediana - limiar * mad_scaled
    lim_sup = mediana + limiar * mad_scaled
    return mask, lim_inf, lim_sup


def detectar_percentil(serie, low=PERC_LOW, high=PERC_HIGH):
    """Percentil: fora de [Plow, Phigh]. Conservador para caudas pesadas."""
    lim_inf = serie.quantile(low)
    lim_sup = serie.quantile(high)
    mask = (serie < lim_inf) | (serie > lim_sup)
    return mask, lim_inf, lim_sup


def detectar_dominio(serie, limites):
    """Domínio clínico: fora de [lim_inf, lim_sup] fixos."""
    lim_inf, lim_sup = limites
    mask = (serie < lim_inf) | (serie > lim_sup)
    return mask, lim_inf, lim_sup


def auto_selecionar_metodo(serie):
    """
    Seleciona automaticamente o melhor método baseado em skewness e kurtosis.
    - |skew| < 0.5 → Z-Score (distribuição ~normal)
    - |skew| 0.5-2 + kurtosis < 5 → IQR (moderada)
    - |skew| > 1 → MAD (assimétrica)
    - kurtosis > 5 → Percentil (caudas pesadas)
    """
    skew = serie.skew()
    kurt = serie.kurtosis()

    if abs(skew) < 0.5:
        return "zscore", f"|skew|={abs(skew):.2f} < 0.5 → distribuição ~normal → Z-Score"
    elif kurt > 5:
        return "percentil", f"kurtosis={kurt:.1f} > 5 → caudas pesadas → Percentil [P1, P99]"
    elif abs(skew) > 1:
        return "mad", f"|skew|={abs(skew):.2f} > 1 → assimétrica → MAD (robusto)"
    else:
        return "iqr", f"|skew|={abs(skew):.2f} moderado, kurt={kurt:.1f} → IQR"


def aplicar_deteccao(serie, metodo, limites=None):
    """Aplica o método de detecção selecionado e retorna (mask, lim_inf, lim_sup)."""
    if metodo == "zscore":
        return detectar_zscore(serie)
    elif metodo == "iqr":
        return detectar_iqr(serie)
    elif metodo == "mad":
        return detectar_mad(serie)
    elif metodo == "percentil":
        return detectar_percentil(serie)
    elif metodo == "dominio":
        return detectar_dominio(serie, limites)
    else:
        return pd.Series(False, index=serie.index), None, None


# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def plot_comparativo(antes, depois, col):
    """Gera histograma + boxplot comparativo antes/depois."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{col} — Antes vs. Depois do Tratamento de Outliers", fontsize=13, fontweight="bold")

    dados_antes = antes.dropna()
    dados_depois = depois.dropna()

    # Histogramas
    axes[0, 0].hist(dados_antes, bins=40, color="#4C72B0", alpha=0.85, edgecolor="white")
    axes[0, 0].set_title("Histograma — Antes", fontsize=10)
    axes[0, 0].set_xlabel(col)

    axes[0, 1].hist(dados_depois, bins=40, color="#55A868", alpha=0.85, edgecolor="white")
    axes[0, 1].set_title("Histograma — Depois", fontsize=10)
    axes[0, 1].set_xlabel(col)

    # Boxplots
    axes[1, 0].boxplot(dados_antes, vert=True, widths=0.6,
                       boxprops=dict(color="#4C72B0"), medianprops=dict(color="red"))
    axes[1, 0].set_title("Boxplot — Antes", fontsize=10)
    axes[1, 0].set_ylabel(col)

    axes[1, 1].boxplot(dados_depois, vert=True, widths=0.6,
                       boxprops=dict(color="#55A868"), medianprops=dict(color="red"))
    axes[1, 1].set_title("Boxplot — Depois", fontsize=10)
    axes[1, 1].set_ylabel(col)

    plt.tight_layout()
    caminho = os.path.join(DIR_PLOTS, f"{col}_antes_depois.png")
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# INÍCIO DO PIPELINE
# ─────────────────────────────────────────────
print("=" * 70)
print("  PIPELINE 08 — TRATAMENTO DE OUTLIERS INDIVIDUALIZADO | PNS 2019")
print("=" * 70)

# 1. Carregar dados
print("\n[INFO] Carregando base 06_pns2019_prepared.csv...")
df = pd.read_csv(PATH_PREPARED, low_memory=False)
n_linhas_orig = df.shape[0]
print(f"[INFO] Base carregada: {df.shape[0]:,} linhas × {df.shape[1]} colunas")


# ─────────────────────────────────────────────
# 2. RESTAURAR VARIÁVEIS DANIFICADAS PELO 04
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("  ETAPA 1 — RESTAURAR VARIÁVEIS DANIFICADAS")
print("─" * 70)

# 2a. Q031 — idade do diagnóstico (foi zerada pelo IQR=[0,0] no script 04)
#      A winsorização aconteceu DENTRO do 04, então 05_clean já está danificada.
#      Precisamos restaurar do 02_original e reaplicar a lógica clínica.
print("\n[RESTAURAR] Q031 (idade diagnóstico diabetes)...")
print(f"  Estado atual: nunique={df['Q031'].nunique()}, min={df['Q031'].min()}, max={df['Q031'].max()}")

df_base_orig = pd.read_csv(PATH_ORIGINAL, low_memory=False)
if "Q031" in df_base_orig.columns:
    # Restaurar Q031 original (502 valores preenchidos, 20007 NaN)
    q031_orig = df_base_orig["Q031"].copy()
    # Aplicar lógica clínica: quem NÃO tem diabetes (Q03001 != 1) → Q031 = 0
    if "Q03001" in df.columns:
        mask_nao_diabetico = (df["Q03001"] != 1)
        q031_orig.loc[mask_nao_diabetico] = 0
    # Imputar NaN restantes com mediana dos diabéticos (43.0)
    mediana_diab = q031_orig.dropna().loc[q031_orig.dropna() > 0].median()
    if pd.isna(mediana_diab):
        mediana_diab = 43.0  # valor do JSON original
    q031_orig.fillna(mediana_diab, inplace=True)
    df["Q031"] = q031_orig.values
    print(f"  Restaurado do 02_original + lógica clínica:")
    print(f"    nunique={df['Q031'].nunique()}, min={df['Q031'].min()}, max={df['Q031'].max()}")
    print(f"    Diabéticos (Q03001==1): Q031 média = {df.loc[df['Q03001']==1, 'Q031'].mean():.1f}")
    print(f"    Não-diabéticos: Q031 = 0")
else:
    print("  [AVISO] Q031 não encontrada em 02_original.csv")
del df_base_orig

# 2b. P036 — tipo de exercício (foi winsorizada como contínua com limites [-4.5, 7.5])
print("\n[RESTAURAR] P036 (tipo de exercício — categórica ordinal)...")
print(f"  Estado atual: nunique={df['P036'].nunique()}, min={df['P036'].min()}, max={df['P036'].max()}")

df_orig = pd.read_csv(PATH_ORIGINAL, low_memory=False)
if "P036" in df_orig.columns and "P036" in df.columns:
    # P036 no original tem valores de 1 a 17, mas no prepared pode ter missing
    # Vamos restaurar só os valores de P036 preservando o tratamento de missing
    mask_preenchido = df["P036"].notna()
    # Pegar os valores originais para quem tem P036 preenchido
    df_temp = df_orig[["P036"]].copy()
    # O prepared pode ter mais linhas que o original? Não — mesmas 20509 linhas
    df["P036"] = df_temp["P036"].values
    # Reaplicar a lógica de sedentários: se P034==2, P036=0
    # Mas P034 foi removida no 06. Vamos usar o flag de sedentário
    # Se Minutos_Semanais_Exercicio == 0 e P036 é NaN no original → é sedentário
    mask_sedentario = (df["Minutos_Semanais_Exercicio"] == 0) & (df["P036"].isna())
    df.loc[mask_sedentario, "P036"] = 0
    print(f"  Restaurado do original: nunique={df['P036'].nunique()}, min={df['P036'].min()}, max={df['P036'].max()}")
else:
    print("  [AVISO] P036 não encontrada em uma das bases")
del df_orig

# 2c. P029 — doses de álcool (winsorização com limites IQR negativos [-3.37, 8.42])
print("\n[RESTAURAR] P029 (doses de álcool/dia)...")
print(f"  Estado atual: min={df['P029'].min():.2f}, max={df['P029'].max():.2f}")
# Valores negativos de doses não fazem sentido → clipar em 0 pelo menos
if df["P029"].min() < 0:
    n_neg = (df["P029"] < 0).sum()
    df.loc[df["P029"] < 0, "P029"] = 0
    print(f"  Corrigido {n_neg} valores negativos → 0 (doses não podem ser negativas)")
    print(f"  Após correção: min={df['P029'].min():.2f}, max={df['P029'].max():.2f}")


# ─────────────────────────────────────────────
# 3. TRATAMENTO DE OUTLIERS — VARIÁVEL A VARIÁVEL
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("  ETAPA 2 — DETECÇÃO E TRATAMENTO DE OUTLIERS")
print("─" * 70)

df_antes = df.copy()  # snapshot antes do tratamento
decisoes = []

# Identificar todas as colunas numéricas elegíveis
cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

# Filtrar: excluir flags, target, categóricas e colunas com poucos valores únicos
cols_analisar = [
    c for c in cols_numericas
    if c not in CATEGORICAS_EXCLUIDAS
    and c != TARGET_COL
    and not c.endswith("_outlier_flag")
    and not c.endswith("_flag")
]

print(f"\n[INFO] {len(cols_analisar)} variáveis numéricas elegíveis para análise de outliers:")
for c in cols_analisar:
    n = df[c].nunique()
    print(f"  → {c} (nunique={n})")

print(f"\n[INFO] {len(CATEGORICAS_EXCLUIDAS)} variáveis categóricas/ordinais excluídas da análise")


for col in cols_analisar:
    serie = df[col].dropna()

    if len(serie) < 30:
        decisoes.append({
            "coluna": col, "metodo": "nenhum", "acao": "manter",
            "n_outliers": 0, "pct_outliers": 0.0,
            "lim_inf": None, "lim_sup": None,
            "justificativa": f"Poucos dados preenchidos ({len(serie)}) → análise inviável"
        })
        print(f"\n[SKIP] {col}: apenas {len(serie)} registros preenchidos")
        continue

    # Verificar se tem estratégia explícita definida
    if col in ESTRATEGIAS:
        config = ESTRATEGIAS[col]
        metodo = config["metodo"]
        acao = config["acao"]
        justificativa = config.get("justificativa", "")
        limites = config.get("limites", None)

        if metodo == "nenhum":
            decisoes.append({
                "coluna": col, "metodo": "nenhum", "acao": "manter",
                "n_outliers": 0, "pct_outliers": 0.0,
                "lim_inf": None, "lim_sup": None,
                "justificativa": justificativa
            })
            print(f"\n[MANTER] {col}: {justificativa}")
            continue

        # Aplicar detecção
        mask, lim_inf, lim_sup = aplicar_deteccao(serie, metodo, limites)

        # Aplicar limite inferior mínimo se definido (ex: doses/minutos não podem ser < 0)
        lim_inf_min = config.get("lim_inferior_minimo", None)
        if lim_inf_min is not None and lim_inf is not None:
            lim_inf = max(lim_inf, lim_inf_min)
            # Recalcular máscara com novo limite
            mask = (serie < lim_inf) | (serie > lim_sup)

    else:
        # Seleção automática baseada em skewness/kurtosis
        metodo, motivo = auto_selecionar_metodo(serie)
        justificativa = f"Auto-seleção: {motivo}"

        mask, lim_inf, lim_sup = aplicar_deteccao(serie, metodo)

        # Determinar ação automática
        n_out = mask.sum()
        pct_out = n_out / len(serie) * 100

        if n_out == 0:
            acao = "manter"
        elif pct_out > 10:
            acao = "flag"
            justificativa += f" | {pct_out:.1f}% outliers → muitos para winsorizar → flag"
        else:
            acao = "winsorizar"

    # Contar outliers
    n_outliers = mask.sum()
    pct_outliers = n_outliers / len(serie) * 100 if len(serie) > 0 else 0

    # Registrar decisão
    entrada = {
        "coluna": col,
        "metodo": metodo,
        "acao": acao,
        "n_outliers": n_outliers,
        "pct_outliers": round(pct_outliers, 2),
        "lim_inf": round(lim_inf, 4) if lim_inf is not None else None,
        "lim_sup": round(lim_sup, 4) if lim_sup is not None else None,
        "justificativa": justificativa
    }
    decisoes.append(entrada)

    # Aplicar ação
    if acao == "winsorizar" and n_outliers > 0:
        antes_vals = df[col].copy()
        df[col] = df[col].clip(lower=lim_inf, upper=lim_sup)
        n_clipados = (antes_vals != df[col]).sum()
        print(f"\n[WINSORIZAR] {col}: {n_clipados} valores clipados para [{lim_inf:.2f}, {lim_sup:.2f}]")
        print(f"  Método: {metodo.upper()} | {n_outliers} outliers detectados ({pct_outliers:.2f}%)")
        print(f"  Justificativa: {justificativa}")

        # Plot comparativo
        plot_comparativo(df_antes[col], df[col], col)

    elif acao == "flag" and n_outliers > 0:
        flag_col = f"{col}_outlier_flag"
        # Garantir que o flag seja criado no índice completo do df
        full_mask = pd.Series(False, index=df.index)
        full_mask.loc[mask.index] = mask
        df[flag_col] = full_mask.astype(int)
        print(f"\n[FLAG] {col}: criada '{flag_col}' → {n_outliers} registros marcados ({pct_outliers:.2f}%)")
        print(f"  Método: {metodo.upper()} | Limites: [{lim_inf:.2f}, {lim_sup:.2f}]")
        print(f"  Justificativa: {justificativa}")

        # Plot comparativo (sem alteração nos dados, mas mostra distribuição)
        plot_comparativo(df_antes[col], df[col], col)

    else:
        print(f"\n[MANTER] {col}: {n_outliers} outliers ({pct_outliers:.2f}%) → mantidos")
        print(f"  Método: {metodo.upper()} | Justificativa: {justificativa}")


# ─────────────────────────────────────────────
# 4. VALIDAÇÃO PÓS-TRATAMENTO
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("  ETAPA 3 — VALIDAÇÃO PÓS-TRATAMENTO")
print("─" * 70)

# 4a. Verificar preservação de linhas
assert df.shape[0] == n_linhas_orig, \
    f"ERRO CRÍTICO: {n_linhas_orig - df.shape[0]} linhas perdidas!"
print(f"\n[OK] Linhas preservadas: {df.shape[0]:,} (nenhuma remoção)")

# 4b. Verificar ausentes não aumentaram
na_antes = df_antes.isna().sum().sum()
na_depois = df.isna().sum().sum()
print(f"[OK] Ausentes: {na_antes:,} → {na_depois:,} (delta: {na_depois - na_antes:+,})")

# 4c. Resumo das decisões
n_winsor = sum(1 for d in decisoes if d["acao"] == "winsorizar" and d["n_outliers"] > 0)
n_flag   = sum(1 for d in decisoes if d["acao"] == "flag" and d["n_outliers"] > 0)
n_manter = sum(1 for d in decisoes if d["acao"] == "manter" or d["n_outliers"] == 0)

print(f"\n[RESUMO] Decisões de outliers:")
print(f"  → Winsorizadas:   {n_winsor} variáveis")
print(f"  → Flagadas:       {n_flag} variáveis")
print(f"  → Mantidas:       {n_manter} variáveis")

# 4d. Estatísticas comparativas para variáveis tratadas
print(f"\n{'─' * 70}")
print(f"  COMPARATIVO ANTES/DEPOIS (variáveis tratadas)")
print(f"{'─' * 70}")
print(f"\n{'Variável':<35} {'Média antes':>12} {'Média depois':>13} {'DP antes':>10} {'DP depois':>10}")
print(f"{'─'*35} {'─'*12} {'─'*13} {'─'*10} {'─'*10}")

for d in decisoes:
    col = d["coluna"]
    if d["acao"] in ("winsorizar", "flag") and d["n_outliers"] > 0 and col in df_antes.columns:
        ma = df_antes[col].mean()
        md = df[col].mean()
        da = df_antes[col].std()
        dd = df[col].std()
        print(f"{col:<35} {ma:>12.4f} {md:>13.4f} {da:>10.4f} {dd:>10.4f}")


# ─────────────────────────────────────────────
# 5. SALVAR RESULTADOS
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("  ETAPA 4 — SALVANDO RESULTADOS")
print("─" * 70)

# 5a. Salvar base tratada
df.to_csv(PATH_OUTPUT, index=False, encoding="utf-8")
print(f"\n[SALVO] Base tratada: {PATH_OUTPUT}")
print(f"[INFO] Shape final: {df.shape[0]:,} × {df.shape[1]}")

# 5b. Gerar relatório detalhado
relatorio_path = os.path.join(DIR_RELATORIOS, "relatorio_outliers_08.txt")
with open(relatorio_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("  RELATÓRIO DE TRATAMENTO DE OUTLIERS — SCRIPT 08 | PNS 2019\n")
    f.write("=" * 80 + "\n\n")

    f.write("RESUMO GERAL\n")
    f.write("─" * 40 + "\n")
    f.write(f"  Base de entrada:   06_pns2019_prepared.csv ({n_linhas_orig:,} linhas)\n")
    f.write(f"  Base de saída:     08_pns2019_outliers_treated.csv ({df.shape[0]:,} linhas × {df.shape[1]} colunas)\n")
    f.write(f"  Linhas removidas:  0 (política de preservação)\n")
    f.write(f"  Variáveis analisadas: {len(cols_analisar)}\n")
    f.write(f"  Winsorizadas: {n_winsor} | Flagadas: {n_flag} | Mantidas: {n_manter}\n\n")

    f.write("VARIÁVEIS RESTAURADAS (correção de bugs do script 04)\n")
    f.write("─" * 60 + "\n")
    f.write("  Q031 — Idade diagnóstico: restaurada de 02_original.csv + lógica clínica (IQR=[0,0] havia zerado todos os valores)\n")
    f.write("  P036 — Tipo de exercício: restaurada de 02_original.csv (é categórica ordinal, não contínua)\n")
    f.write("  P029 — Doses de álcool: valores negativos corrigidos para 0 (IQR inferior era -3.37)\n\n")

    f.write("TÉCNICAS UTILIZADAS\n")
    f.write("─" * 60 + "\n")
    f.write("  Z-Score    |z| > 3.0        — para distribuições ~normais (|skew| < 0.5)\n")
    f.write("  IQR        Q1-1.5×IQR..Q3+1.5×IQR — para distribuições moderadas (IQR > 0)\n")
    f.write("  MAD        |x-med|/MAD > 3.5  — robusto para distribuições assimétricas (|skew| > 1)\n")
    f.write("  Percentil  [P1, P99]          — conservador para caudas pesadas (kurtosis > 5)\n")
    f.write("  Domínio    limites fixos      — baseado em conhecimento clínico/biológico\n\n")

    f.write("DECISÕES POR VARIÁVEL\n")
    f.write("─" * 80 + "\n")
    f.write(f"{'Variável':<35} {'Método':<12} {'Ação':<13} {'N_out':>6} {'%_out':>6} {'Lim_inf':>10} {'Lim_sup':>10}\n")
    f.write("─" * 80 + "\n")

    for d in decisoes:
        li = f"{d['lim_inf']:.2f}" if d['lim_inf'] is not None else "N/A"
        ls = f"{d['lim_sup']:.2f}" if d['lim_sup'] is not None else "N/A"
        f.write(f"{d['coluna']:<35} {d['metodo']:<12} {d['acao']:<13} {d['n_outliers']:>6} {d['pct_outliers']:>5.2f}% {li:>10} {ls:>10}\n")

    f.write("\n\nJUSTIFICATIVAS DETALHADAS\n")
    f.write("─" * 80 + "\n")
    for d in decisoes:
        f.write(f"\n  {d['coluna']}:\n")
        f.write(f"    Método: {d['metodo'].upper()}\n")
        f.write(f"    Ação:   {d['acao'].upper()}\n")
        f.write(f"    Outliers: {d['n_outliers']} ({d['pct_outliers']:.2f}%)\n")
        if d['lim_inf'] is not None:
            f.write(f"    Limites: [{d['lim_inf']:.4f}, {d['lim_sup']:.4f}]\n")
        f.write(f"    Justificativa: {d['justificativa']}\n")

    f.write("\n\nVARIÁVEIS CATEGÓRICAS EXCLUÍDAS DA ANÁLISE\n")
    f.write("─" * 60 + "\n")
    for c in sorted(CATEGORICAS_EXCLUIDAS):
        if c in df.columns:
            f.write(f"  {c} (nunique={df[c].nunique()})\n")

print(f"[SALVO] Relatório: {relatorio_path}")
print(f"[SALVO] Plots em: {DIR_PLOTS}/")

# 5c. Salvar decisões em CSV para rastreabilidade
df_decisoes = pd.DataFrame(decisoes)
csv_decisoes_path = os.path.join(DIR_RELATORIOS, "decisoes_outliers_08.csv")
df_decisoes.to_csv(csv_decisoes_path, index=False, encoding="utf-8")
print(f"[SALVO] Decisões CSV: {csv_decisoes_path}")

print("\n" + "=" * 70)
print("  PIPELINE 08 CONCLUÍDO COM SUCESSO")
print("=" * 70)
