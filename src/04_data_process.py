"""
Script: 04_tratamento_ausentes_outliers.py
Objetivo: Tratamento inteligente de dados ausentes e outliers baseado nas
          métricas pré-calculadas por variável (JSON), seguindo os mecanismos
          MCAR/MAR/NMAR e as diretrizes do Prof. Zárate (Slides 12, 13, 14).
Input:
    data/processed/03_pns2019_column_metrics.json  ← métricas por variável
    data/processed/02_pns2019_selected_columns.csv ← base de dados
Output:
    data/processed/04_pns2019_missing_treated.csv  ← após tratamento de ausentes
    data/processed/05_pns2019_clean.csv            ← após tratamento de outliers
    docs/relatorios/decisoes_ausentes.csv          ← log de decisões de imputação
    docs/relatorios/decisoes_outliers.csv          ← log de decisões de outliers
    docs/relatorios/plots/                         ← histogramas e boxplots antes/depois
Dependências: pandas, numpy, scipy, sklearn, matplotlib, seaborn, statsmodels
Referências:
    - Prof. Zárate — Slide 12 (Preparação de Dados)
    - Prof. Zárate — Slide 13 (Análise de Dados Ausentes)
    - Prof. Zárate — Slide 14 (Análise de Outliers)
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_JSON    = os.path.join(BASE_DIR, "data", "processed", "03_pns2019_column_metrics.json")
PATH_CSV     = os.path.join(BASE_DIR, "data", "processed", "02_pns2019_selected_columns.csv")
PATH_OUT_MISSING  = os.path.join(BASE_DIR, "data", "processed", "04_pns2019_missing_treated.csv")
PATH_OUT_CLEAN    = os.path.join(BASE_DIR, "data", "processed", "05_pns2019_clean.csv")
DIR_RELATORIOS    = os.path.join(BASE_DIR, "docs", "relatorios")
DIR_PLOTS         = os.path.join(BASE_DIR, "docs", "relatorios", "plots")

os.makedirs(DIR_RELATORIOS, exist_ok=True)
os.makedirs(DIR_PLOTS, exist_ok=True)

# Variável-alvo — diagnóstico de diabetes
TARGET_COL = "Q03001"

# Thresholds
THRESH_ALTA_AUSENCIA   = 0.70   # >70% ausente → remover coluna
THRESH_KNN             = 0.05   # entre 5% e 30% → KNN
THRESH_REGRESSAO       = 0.30   # >30% → regressão
THRESH_SKEW_NORMAL     = 0.5    # |skewness| < 0.5 → distribuição aproximadamente normal
THRESH_CORR_REDUNDANTE = 0.95   # correlação > 0.95 → redundante
ZSCORE_LIMIAR          = 3.0    # |z| > 3 → outlier
IQR_MULTIPLIER         = 1.5    # IQR × 1.5 → limites boxplot
CONTAMINATION_IF       = 0.05   # Isolation Forest: 5% de anomalias esperadas
KNN_K                  = 5      # vizinhos para KNN imputer

# Variáveis estruturalmente vazias por domínio clínico (ausente ≠ vazio)
# Adicione aqui variáveis que NÃO devem ser imputadas para certos grupos
# Formato: {coluna: (coluna_filtro, valor_que_torna_vazio)}
VAZIOS_ESTRUTURAIS = {
    # Exemplo: "Q121" só faz sentido para mulheres (sexo == 1)
    # "Q121": ("C006", 2),
}

print("=" * 65)
print("  PIPELINE DE TRATAMENTO — AUSENTES E OUTLIERS | PNS 2019")
print("=" * 65)


# ─────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────
print("\n[INFO] Carregando arquivos...")

with open(PATH_JSON, "r", encoding="utf-8") as f:
    metricas = json.load(f)

df_orig = pd.read_csv(PATH_CSV, low_memory=False)
df = df_orig.copy()

print(f"[INFO] Base carregada: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
print(f"[INFO] Métricas carregadas: {len(metricas)} variáveis no JSON")

# Verifica cobertura do JSON
cols_sem_metrica = [c for c in df.columns if c not in metricas]
if cols_sem_metrica:
    print(f"[AVISO] {len(cols_sem_metrica)} colunas no CSV sem entrada no JSON: {cols_sem_metrica}")


# ─────────────────────────────────────────────
# 2. DETECÇÃO AUTOMÁTICA DO TARGET
# ─────────────────────────────────────────────
if TARGET_COL is None:
    # Heurística: variável binária com nome relacionado a diabetes/doença
    candidatos_target = [
        c for c, m in metricas.items()
        if m.get("tipo") in ("binario", "categorico")
        and m.get("n_distinct", 99) == 2
        and c in df.columns
    ]
    if candidatos_target:
        TARGET_COL = candidatos_target[0]
        print(f"[INFO] Target detectado automaticamente: {TARGET_COL} "
              f"(ajuste TARGET_COL manualmente se incorreto)")
    else:
        print("[AVISO] Nenhum target detectado. Defina TARGET_COL manualmente.")


# ─────────────────────────────────────────────
# 3. CLASSIFICAÇÃO DAS VARIÁVEIS VIA JSON
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 2.5 TRATAMENTO CLÍNICO ESPECÍFICO (VAZIOS ESTRUTURAIS)
# ─────────────────────────────────────────────
print("\n[INFO] Aplicando correções de domínio (vazios estruturais)...")

# Q03001: Algum médico já lhe deu diagnóstico de diabetes? (1=Sim)
# Q031: Idade no primeiro diagnóstico. Se não tem diabetes, imputamos 0.
if "Q031" in df.columns and "Q03001" in df.columns:
    df.loc[df["Q03001"] != 1, "Q031"] = 0

# Exercício Físico (P034: 1=Sim, 2=Não)
if "P034" in df.columns:
    mask_sedentario = (df["P034"] == 2)
    for c in ["P035", "P036", "P03701", "P03702"]:
        if c in df.columns: df.loc[mask_sedentario, c] = 0

# Deslocamento para o trabalho (P040: 1=Sim, 2=Não)
if "P040" in df.columns:
    mask_sem_desloc = (df["P040"] == 2)
    for c in ["P04001", "P04101", "P04102"]:
        if c in df.columns: df.loc[mask_sem_desloc, c] = 0

# Bebida Alcoólica (P027: 4=Nunca)
if "P027" in df.columns:
    mask_abstemio = (df["P027"] == 4)
    if "P02801" in df.columns: df.loc[mask_abstemio, "P02801"] = 0
    if "P029" in df.columns: df.loc[mask_abstemio, "P029"] = 0
    if "P03201" in df.columns: df.loc[mask_abstemio, "P03201"] = 2 # 2=Não

# Verduras e Frutas
if "P00901" in df.columns and "P01001" in df.columns:
    df.loc[df["P00901"] == 0, "P01001"] = 0
if "P018" in df.columns and "P019" in df.columns:
    df.loc[df["P018"] == 0, "P019"] = 0

# Tabagismo
if "P050" in df.columns and "P05401" in df.columns:
    df.loc[df["P050"] == 2, "P05401"] = 0

print("\n[INFO] Classificando variáveis com base no JSON de métricas...")

# Mapeamento de tipo do JSON → categoria interna
TIPO_CONTINUO   = {"numerico_continuo"}
TIPO_DISCRETO   = {"numerico_discreto", "numerico_inteiro"}
TIPO_BINARIO    = {"binario"}
TIPO_CATEGORICO = {"categorico", "nominal", "ordinal"}

decisoes_ausentes = []  # log de decisões

cols_remover       = []
cols_continuas     = []
cols_discretas     = []
cols_binarias      = []
cols_categoricas   = []
cols_sem_ausencia  = []

for col, m in metricas.items():
    if col not in df.columns:
        continue

    tipo       = m.get("tipo", "desconhecido")
    
    # RECALCULA MÉTRICAS APÓS TRATAMENTO CLÍNICO
    ausentes   = df[col].isna().sum()
    pct        = ausentes / len(df)

    skewness   = m.get("skewness", 0.0) or 0.0
    media      = m.get("media")
    mediana    = m.get("mediana")
    dp         = m.get("desvio_padrao")
    n_distinct = m.get("n_distinct", 0)

    # --- Decisão de remoção ---
    if pct > THRESH_ALTA_AUSENCIA:
        cols_remover.append(col)
        decisoes_ausentes.append({
            "coluna": col, "tipo": tipo, "pct_ausentes": round(pct * 100, 2),
            "mecanismo": "N/A", "estrategia": "REMOVER_COLUNA",
            "justificativa": f"{pct*100:.1f}% ausentes > limiar de {THRESH_ALTA_AUSENCIA*100:.0f}%"
        })
        print(f"[REMOVIDO] {col}: {pct*100:.1f}% ausentes → coluna removida")
        continue

    # --- Variável constante (n_distinct == 1) ---
    if n_distinct == 1:
        cols_remover.append(col)
        decisoes_ausentes.append({
            "coluna": col, "tipo": tipo, "pct_ausentes": round(pct * 100, 2),
            "mecanismo": "N/A", "estrategia": "REMOVER_CONSTANTE",
            "justificativa": "Variável constante (n_distinct=1), inútil para ML"
        })
        print(f"[REMOVIDO] {col}: constante (n_distinct=1)")
        continue

    # --- Sem ausência ---
    if ausentes == 0:
        cols_sem_ausencia.append(col)
        continue

    # --- Classificação por tipo ---
    if tipo in TIPO_CONTINUO:
        cols_continuas.append((col, pct, skewness, media, mediana, dp))
    elif tipo in TIPO_DISCRETO:
        cols_discretas.append((col, pct, skewness, media, mediana, dp))
    elif tipo in TIPO_BINARIO:
        cols_binarias.append((col, pct))
    elif tipo in TIPO_CATEGORICO:
        cols_categoricas.append((col, pct))
    else:
        # Fallback: tenta inferir pelo n_distinct
        if n_distinct <= 2:
            cols_binarias.append((col, pct))
        elif n_distinct <= 15:
            cols_categoricas.append((col, pct))
        else:
            cols_continuas.append((col, pct, skewness, media, mediana, dp))

print(f"\n[INFO] Resumo da classificação:")
print(f"  → Remover:              {len(cols_remover)} variáveis")
print(f"  → Sem ausência:         {len(cols_sem_ausencia)} variáveis")
print(f"  → Contínuas c/ missing: {len(cols_continuas)} variáveis")
print(f"  → Discretas c/ missing: {len(cols_discretas)} variáveis")
print(f"  → Binárias c/ missing:  {len(cols_binarias)} variáveis")
print(f"  → Categóricas c/ miss.: {len(cols_categoricas)} variáveis")


# ─────────────────────────────────────────────
# 4. IDENTIFICAÇÃO DO MECANISMO DE AUSÊNCIA
# ─────────────────────────────────────────────
def identificar_mecanismo(df, col, target_col, limiar_pvalor=0.05):
    """
    Testa MCAR vs MAR usando regressão logística.
    Cria dummy: ausente=1, presente=0. Se alguma variável prediz
    a ausência com significância, classifica como MAR. Caso contrário,
    MCAR. NMAR é avaliado pela teoria (não testável estatisticamente).

    Retorna: 'MCAR' | 'MAR' | 'NMAR_suspeito'
    """
    dummy = df[col].isna().astype(int)
    if dummy.sum() < 10:
        return "MCAR"  # poucos ausentes, sem poder estatístico

    # Preditoras: numéricas sem ausência
    preditoras = [
        c for c in df.columns
        if c != col
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        and df[c].notna().all()
    ][:20]  # limita para performance

    if len(preditoras) < 2:
        return "MCAR"

    try:
        X = df[preditoras].values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
        lr.fit(X_sc, dummy)
        # Usa p-valor via statsmodels para coeficientes
        from statsmodels.discrete.discrete_model import Logit
        import statsmodels.api as sm
        X_sm = sm.add_constant(X_sc)
        model = Logit(dummy, X_sm).fit(disp=False)
        pvalores = model.pvalues[1:]  # exclui constante
        if (pvalores < limiar_pvalor).any():
            return "MAR"
        else:
            return "MCAR"
    except Exception:
        return "MCAR"  # fallback conservador


# ─────────────────────────────────────────────
# 5. FUNÇÕES AUXILIARES DE IMPUTAÇÃO
# ─────────────────────────────────────────────

def plot_comparativo(df_antes, df_depois, col, tipo="hist", subdir=""):
    """Salva histograma ou boxplot antes/depois da imputação."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{col} — antes vs. depois", fontsize=12)

    dados_antes  = df_antes[col].dropna()
    dados_depois = df_depois[col].dropna()

    if tipo == "hist":
        axes[0].hist(dados_antes,  bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
        axes[1].hist(dados_depois, bins=30, color="#55A868", alpha=0.8, edgecolor="white")
    else:
        axes[0].boxplot(dados_antes,  vert=True)
        axes[1].boxplot(dados_depois, vert=True)

    axes[0].set_title("Antes")
    axes[1].set_title("Depois")
    for ax in axes:
        ax.set_xlabel(col)

    pasta = os.path.join(DIR_PLOTS, subdir)
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, f"{col}_{tipo}.png")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()


def validar_variancia(col, serie_antes, serie_depois, threshold=0.05):
    """
    Verifica se o desvio padrão caiu mais que threshold (default 5%).
    Retorna (passou, dp_antes, dp_depois, queda_pct).
    Ref: Zárate Slide 13 — Imputação Não Polarizada.
    """
    dp_antes  = serie_antes.std()
    dp_depois = serie_depois.std()
    if dp_antes == 0:
        return True, dp_antes, dp_depois, 0.0
    queda = (dp_antes - dp_depois) / dp_antes
    passou = queda <= threshold
    return passou, dp_antes, dp_depois, queda


def imputar_com_ruido(df, col, dp_original):
    """
    Imputação estocástica: preenche NaN com mediana + ruído gaussiano
    (média=0, dp=dp_original×0.1) para preservar variância.
    Ref: Zárate Slide 13 — Imputação Não Polarizada.
    """
    mediana = df[col].median()
    n_ausentes = df[col].isna().sum()
    ruido = np.random.normal(0, dp_original * 0.1, n_ausentes)
    df.loc[df[col].isna(), col] = mediana + ruido
    return df


def obter_top_correlatas(df, col, n=5):
    """Retorna as N colunas numéricas mais correlacionadas com col."""
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if col in numericas:
        numericas.remove(col)
    corrs = df[numericas].corrwith(df[col]).abs().dropna().sort_values(ascending=False)
    return corrs.head(n).index.tolist()


# ─────────────────────────────────────────────
# 6. TRATAMENTO DE DADOS AUSENTES
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ETAPA 1 — TRATAMENTO DE DADOS AUSENTES")
print("=" * 65)

# Remove colunas marcadas para exclusão
df.drop(columns=[c for c in cols_remover if c in df.columns], inplace=True)
print(f"\n[INFO] {len(cols_remover)} colunas removidas por alta ausência ou constância")

df_pre_missing = df.copy()  # snapshot antes da imputação


def tratar_continua(df, col, pct, skewness, media, mediana, dp):
    """
    Lógica de decisão para variáveis contínuas:
      <5% + normal (|skew|<0.5) → média
      <5% + assimétrica         → mediana
      5–30%                     → KNN
      >30%                      → regressão linear
    Sempre valida preservação de variância.
    """
    mecanismo = identificar_mecanismo(df, col, TARGET_COL)
    entrada   = {"coluna": col, "tipo": "continua", "pct_ausentes": round(pct * 100, 2),
                 "mecanismo": mecanismo}

    # NMAR: cria dummy e não imputa
    if mecanismo == "NMAR_suspeito":
        flag_col = f"{col}_missing_flag"
        df[flag_col] = df[col].isna().astype(int)
        entrada.update({"estrategia": "DUMMY_NMAR",
                        "justificativa": f"Mecanismo NMAR suspeito → criada flag '{flag_col}'"})
        print(f"[AVISO] {col}: NMAR suspeito → flag criada, sem imputação automática")
        decisoes_ausentes.append(entrada)
        return df

    serie_antes = df[col].copy()

    if pct < THRESH_KNN:  # <5%
        if abs(skewness) < THRESH_SKEW_NORMAL:
            valor = media if media is not None else df[col].mean()
            df[col] = df[col].fillna(valor)
            estrategia = f"MEDIA ({valor:.4f})"
            just = f"|skewness|={abs(skewness):.2f} < {THRESH_SKEW_NORMAL} → dist. normal → média"
        else:
            valor = mediana if mediana is not None else df[col].median()
            df[col] = df[col].fillna(valor)
            estrategia = f"MEDIANA ({valor:.4f})"
            just = f"|skewness|={abs(skewness):.2f} ≥ {THRESH_SKEW_NORMAL} → dist. assimétrica → mediana"

        # Verifica preservação de variância (Zárate Slide 13)
        passou, dp_a, dp_d, queda = validar_variancia(col, serie_antes.dropna(), df[col])
        if not passou:
            print(f"[AVISO] {col}: variância caiu {queda*100:.1f}% → aplicando imputação estocástica")
            dp_ref = dp if dp else serie_antes.std()
            df = imputar_com_ruido(df, col, dp_ref)
            estrategia += " + RUÍDO_GAUSSIANO"

    elif pct <= THRESH_REGRESSAO:  # 5–30% → KNN
        cols_ref = obter_top_correlatas(df, col, n=KNN_K)
        cols_usar = [col] + [c for c in cols_ref if df[c].notna().all()]
        imputer = KNNImputer(n_neighbors=KNN_K)
        df[cols_usar] = imputer.fit_transform(df[cols_usar])
        estrategia = f"KNN (k={KNN_K})"
        just = f"{pct*100:.1f}% ausentes entre 5–30% → KNN com vizinhos: {cols_ref}"

    else:  # >30% → regressão linear
        cols_ref = obter_top_correlatas(df, col, n=8)
        mask_presente = df[col].notna()
        X_cols = [c for c in cols_ref if df[c].notna().all()]
        if len(X_cols) >= 2 and mask_presente.sum() > 50:
            reg = LinearRegression()
            reg.fit(df.loc[mask_presente, X_cols], df.loc[mask_presente, col])
            preditos = reg.predict(df.loc[~mask_presente, X_cols])
            df.loc[~mask_presente, col] = preditos
            estrategia = "REGRESSAO_LINEAR"
            just = f"{pct*100:.1f}% ausentes > 30% → regressão com {X_cols}"
        else:
            valor = df[col].median()
            df[col] = df[col].fillna(valor)
            estrategia = f"MEDIANA_FALLBACK ({valor:.4f})"
            just = "Regressão inviável (poucos preditores ou registros) → mediana"

    entrada.update({"estrategia": estrategia, "justificativa": just})
    decisoes_ausentes.append(entrada)
    print(f"[IMPUTADO] {col}: {estrategia} | {mecanismo} | {pct*100:.1f}% ausentes")

    # Plot comparativo
    plot_comparativo(df_pre_missing, df, col, tipo="hist", subdir="ausentes")
    return df


def tratar_discreta(df, col, pct, skewness, media, mediana, dp):
    """Discreta → mediana arredondada para inteiro mais próximo (Zárate Slide 13)."""
    mecanismo = identificar_mecanismo(df, col, TARGET_COL)
    valor = int(round(mediana)) if mediana is not None else int(round(df[col].median()))
    df[col] = df[col].fillna(valor)
    entrada = {
        "coluna": col, "tipo": "discreta", "pct_ausentes": round(pct * 100, 2),
        "mecanismo": mecanismo, "estrategia": f"MEDIANA_INT ({valor})",
        "justificativa": "Variável discreta → mediana arredondada para inteiro"
    }
    decisoes_ausentes.append(entrada)
    print(f"[IMPUTADO] {col}: MEDIANA_INT={valor} | {mecanismo} | {pct*100:.1f}% ausentes")
    plot_comparativo(df_pre_missing, df, col, tipo="hist", subdir="ausentes")
    return df


def tratar_binaria(df, col, pct):
    """Binária → moda."""
    mecanismo = identificar_mecanismo(df, col, TARGET_COL)
    moda = df[col].mode()
    valor = moda.iloc[0] if not moda.empty else 0
    df[col] = df[col].fillna(valor)
    entrada = {
        "coluna": col, "tipo": "binaria", "pct_ausentes": round(pct * 100, 2),
        "mecanismo": mecanismo, "estrategia": f"MODA ({valor})",
        "justificativa": "Variável binária → moda"
    }
    decisoes_ausentes.append(entrada)
    print(f"[IMPUTADO] {col}: MODA={valor} | {mecanismo} | {pct*100:.1f}% ausentes")
    return df


def tratar_categorica(df, col, pct):
    """Categórica → moda."""
    mecanismo = identificar_mecanismo(df, col, TARGET_COL)
    moda = df[col].mode()
    valor = moda.iloc[0] if not moda.empty else "desconhecido"
    df[col] = df[col].fillna(valor)
    entrada = {
        "coluna": col, "tipo": "categorica", "pct_ausentes": round(pct * 100, 2),
        "mecanismo": mecanismo, "estrategia": f"MODA ({valor})",
        "justificativa": "Variável categórica → moda"
    }
    decisoes_ausentes.append(entrada)
    print(f"[IMPUTADO] {col}: MODA={valor} | {mecanismo} | {pct*100:.1f}% ausentes")
    return df


# Aplica tratamentos
print("\n[INFO] Tratando variáveis contínuas...")
for col, pct, skewness, media, mediana, dp in cols_continuas:
    if col in df.columns:
        df = tratar_continua(df, col, pct, skewness, media, mediana, dp)

print("\n[INFO] Tratando variáveis discretas...")
for col, pct, skewness, media, mediana, dp in cols_discretas:
    if col in df.columns:
        df = tratar_discreta(df, col, pct, skewness, media, mediana, dp)

print("\n[INFO] Tratando variáveis binárias...")
for col, pct in cols_binarias:
    if col in df.columns:
        df = tratar_binaria(df, col, pct)

print("\n[INFO] Tratando variáveis categóricas...")
for col, pct in cols_categoricas:
    if col in df.columns:
        df = tratar_categorica(df, col, pct)

# Verifica se ainda restam ausentes
ausentes_restantes = df.isna().sum()
ausentes_restantes = ausentes_restantes[ausentes_restantes > 0]
if not ausentes_restantes.empty:
    print(f"\n[AVISO] {len(ausentes_restantes)} variáveis ainda com ausentes (flags NMAR ou colunas novas):")
    for c, n in ausentes_restantes.items():
        print(f"  {c}: {n} ausentes")

# Salva log de decisões
df_decisoes_ausentes = pd.DataFrame(decisoes_ausentes)
df_decisoes_ausentes.to_csv(
    os.path.join(DIR_RELATORIOS, "decisoes_ausentes.csv"), index=False, encoding="utf-8"
)

# Salva base após missing
df.to_csv(PATH_OUT_MISSING, index=False, encoding="utf-8")
print(f"\n[INFO] Base pós-missing salva: {PATH_OUT_MISSING}")
print(f"[INFO] Shape: {df.shape[0]:,} × {df.shape[1]}")


# ─────────────────────────────────────────────
# 7. TRATAMENTO DE OUTLIERS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ETAPA 2 — TRATAMENTO DE OUTLIERS")
print("=" * 65)

df_pre_outlier = df.copy()
decisoes_outliers = []

# Apenas variáveis numéricas contínuas (exclui binárias e flags)
cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
cols_outlier = [
    c for c in cols_num
    if c != TARGET_COL
    and not c.endswith("_missing_flag")
    and not c.endswith("_outlier_flag")
    and df[c].nunique() > 10  # exclui discretas com poucos valores
]

print(f"\n[INFO] Analisando outliers em {len(cols_outlier)} variáveis numéricas contínuas...")


def detectar_outliers_univariado(serie):
    """
    Aplica Z-Score (μ±3σ) e IQR simultaneamente.
    Retorna máscaras booleanas e limites.
    Ref: Zárate Slide 14 — Detecção Univariada.
    """
    media = serie.mean()
    dp    = serie.std()
    q1    = serie.quantile(0.25)
    q3    = serie.quantile(0.75)
    iqr   = q3 - q1

    lim_inf_iqr = q1 - IQR_MULTIPLIER * iqr
    lim_sup_iqr = q3 + IQR_MULTIPLIER * iqr

    mask_zscore = (np.abs((serie - media) / (dp + 1e-9)) > ZSCORE_LIMIAR)
    mask_iqr    = (serie < lim_inf_iqr) | (serie > lim_sup_iqr)
    mask_consenso = mask_zscore & mask_iqr  # ambos concordam

    return {
        "mask_zscore":    mask_zscore,
        "mask_iqr":       mask_iqr,
        "mask_consenso":  mask_consenso,
        "lim_inf_iqr":    lim_inf_iqr,
        "lim_sup_iqr":    lim_sup_iqr,
        "lim_inf_zscore": media - ZSCORE_LIMIAR * dp,
        "lim_sup_zscore": media + ZSCORE_LIMIAR * dp,
        "media": media, "dp": dp, "q1": q1, "q3": q3, "iqr": iqr
    }


def decidir_outlier(col, serie, resultado):
    """
    Regra de decisão baseada em Zárate Slide 14:
    - Consenso Z+IQR sem explicação clínica → Winsorização
    - Apenas 1 critério (outlier suave) → manter
    - Cria flag para outliers clinicamente relevantes
    Retorna (estrategia, justificativa)
    """
    n_consenso = resultado["mask_consenso"].sum()
    pct_consenso = n_consenso / len(serie) * 100

    if n_consenso == 0:
        return "MANTER", "Sem outliers confirmados (consenso Z+IQR)"

    # Heurística para outliers clinicamente plausíveis em saúde
    # Variáveis como glicemia, IMC, pressão têm outliers reais (diabéticos)
    VARS_CLINICAS_REAIS = ["P00103", "P00104", "P00403", "P00404", "VDF003", "J012", "E017"]

    if col in VARS_CLINICAS_REAIS:
        return "FLAG", f"Outlier clínico real → flag criada (não winsorizado)"

    # Se >10% de outliers confirmados, winsorização seria muito agressiva
    if pct_consenso > 10:
        return "FLAG", f"{pct_consenso:.1f}% outliers → provavelmente eventos reais → flag"

    return "WINSORIZAR", (
        f"{n_consenso} outliers confirmados ({pct_consenso:.2f}%) → "
        f"winsorização nos limites IQR [{resultado['lim_inf_iqr']:.2f}, {resultado['lim_sup_iqr']:.2f}]"
    )


for col in cols_outlier:
    if col not in df.columns:
        continue

    serie = df[col].dropna()
    if len(serie) < 30:
        continue

    resultado = detectar_outliers_univariado(serie)
    estrategia, justificativa = decidir_outlier(col, serie, resultado)

    n_zscore   = resultado["mask_zscore"].sum()
    n_iqr      = resultado["mask_iqr"].sum()
    n_consenso = resultado["mask_consenso"].sum()

    entrada = {
        "coluna": col,
        "n_zscore": n_zscore, "pct_zscore": round(n_zscore / len(serie) * 100, 2),
        "n_iqr": n_iqr,       "pct_iqr": round(n_iqr / len(serie) * 100, 2),
        "n_consenso": n_consenso,
        "pct_consenso": round(n_consenso / len(serie) * 100, 2),
        "lim_inf_iqr": round(resultado["lim_inf_iqr"], 4),
        "lim_sup_iqr": round(resultado["lim_sup_iqr"], 4),
        "estrategia": estrategia,
        "justificativa": justificativa
    }

    if estrategia == "WINSORIZAR":
        lim_inf = resultado["lim_inf_iqr"]
        lim_sup = resultado["lim_sup_iqr"]

        # Winsorização conservadora: compara com percentis 1/99
        p1  = serie.quantile(0.01)
        p99 = serie.quantile(0.99)
        lim_inf_final = max(lim_inf, p1)   # mais conservador = maior dos dois
        lim_sup_final = min(lim_sup, p99)  # mais conservador = menor dos dois

        n_antes = ((df[col] < lim_inf_final) | (df[col] > lim_sup_final)).sum()
        df[col] = df[col].clip(lower=lim_inf_final, upper=lim_sup_final)
        print(f"[OUTLIER] {col}: WINSORIZADO → {n_antes} valores clipados "
              f"[{lim_inf_final:.2f}, {lim_sup_final:.2f}]")

    elif estrategia == "FLAG":
        flag_col = f"{col}_outlier_flag"
        df[flag_col] = resultado["mask_consenso"].reindex(df.index).fillna(False).astype(int)
        print(f"[OUTLIER] {col}: FLAG criada '{flag_col}' → {n_consenso} registros marcados")

    else:
        print(f"[OUTLIER] {col}: MANTIDO → {n_consenso} outliers confirmados (suave ou nenhum)")

    decisoes_outliers.append(entrada)

    # Plot boxplot comparativo
    plot_comparativo(df_pre_outlier, df, col, tipo="hist", subdir="outliers")


# ─────────────────────────────────────────────
# 8. DETECÇÃO MULTIVARIADA (Mahalanobis + Isolation Forest)
# ─────────────────────────────────────────────
print("\n[INFO] Detecção multivariada de outliers...")

cols_mv = [c for c in cols_outlier if c in df.columns and df[c].notna().all()][:30]

if len(cols_mv) >= 3:
    X_mv = df[cols_mv].values
    scaler_mv = StandardScaler()
    X_sc = scaler_mv.fit_transform(X_mv)

    # Isolation Forest (Zárate Slide 14 — Clusterização)
    iso = IsolationForest(contamination=CONTAMINATION_IF, random_state=RANDOM_STATE)
    pred_iso = iso.fit_predict(X_sc)
    n_anomalias_iso = (pred_iso == -1).sum()
    df["_outlier_isolation_forest"] = (pred_iso == -1).astype(int)
    print(f"[INFO] Isolation Forest: {n_anomalias_iso} anomalias detectadas "
          f"({n_anomalias_iso/len(df)*100:.1f}%)")

    # Distância de Mahalanobis (Zárate Slide 14 — Distância Global)
    try:
        cov    = np.cov(X_sc.T)
        inv_cov = np.linalg.pinv(cov)
        mean_v  = X_sc.mean(axis=0)
        diff    = X_sc - mean_v
        dist_mah = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
        limiar_mah = np.sqrt(chi2.ppf(0.999, df=len(cols_mv)))
        n_mah = (dist_mah > limiar_mah).sum()
        df["_outlier_mahalanobis"] = (dist_mah > limiar_mah).astype(int)
        print(f"[INFO] Mahalanobis: {n_mah} outliers "
              f"(limiar={limiar_mah:.2f}, {n_mah/len(df)*100:.1f}%)")
    except np.linalg.LinAlgError:
        print("[AVISO] Mahalanobis: matriz singular, skipping")
else:
    print("[AVISO] Poucas variáveis numéricas completas para análise multivariada")


# ─────────────────────────────────────────────
# 9. SALVAR OUTPUTS FINAIS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SALVANDO RESULTADOS")
print("=" * 65)

# Remove colunas auxiliares de detecção multivariada (flags de diagnóstico)
cols_aux = [c for c in df.columns if c.startswith("_outlier_")]
df_clean = df.drop(columns=cols_aux, errors="ignore")

# Salva base limpa
df_clean.to_csv(PATH_OUT_CLEAN, index=False, encoding="utf-8")
print(f"\n[INFO] Base limpa salva: {PATH_OUT_CLEAN}")
print(f"[INFO] Shape final: {df_clean.shape[0]:,} × {df_clean.shape[1]}")

# Salva log de outliers
df_decisoes_outliers = pd.DataFrame(decisoes_outliers)
df_decisoes_outliers.to_csv(
    os.path.join(DIR_RELATORIOS, "decisoes_outliers.csv"), index=False, encoding="utf-8"
)


# ─────────────────────────────────────────────
# 10. RELATÓRIO COMPARATIVO FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RELATÓRIO COMPARATIVO FINAL")
print("=" * 65)

df_base = pd.read_csv(PATH_CSV, low_memory=False)
df_miss = pd.read_csv(PATH_OUT_MISSING, low_memory=False)
df_fin  = pd.read_csv(PATH_OUT_CLEAN,   low_memory=False)

relatorio_linhas = []

def resumir(df_, nome):
    aus = df_.isna().sum().sum()
    return {
        "base": nome,
        "linhas": df_.shape[0],
        "colunas": df_.shape[1],
        "total_ausentes": aus,
        "pct_ausentes": round(aus / (df_.shape[0] * df_.shape[1]) * 100, 4)
    }

comparativo = pd.DataFrame([
    resumir(df_base, "01_original"),
    resumir(df_miss, "04_pos_missing"),
    resumir(df_fin,  "05_pos_outliers"),
])

print(comparativo.to_string(index=False))

# KS test para verificar preservação de distribuição
print("\n[INFO] Teste Kolmogorov-Smirnov — distribuição preservada? (p>0.05 = sim)")
cols_ks = [c for c in cols_outlier if c in df_base.columns and c in df_fin.columns][:10]
for col in cols_ks:
    serie_orig = df_base[col].dropna()
    serie_fin  = df_fin[col].dropna()
    if len(serie_orig) > 10 and len(serie_fin) > 10:
        stat, p = stats.ks_2samp(serie_orig, serie_fin)
        ok = "✓ PRESERVADA" if p > 0.05 else "⚠ ALTERADA"
        print(f"  {col}: KS={stat:.4f}, p={p:.4f} → {ok}")

# Resumo de decisões
n_removidas  = len([d for d in decisoes_ausentes if "REMOVER" in d.get("estrategia", "")])
n_imputadas  = len([d for d in decisoes_ausentes if "REMOVER" not in d.get("estrategia", "") and "DUMMY" not in d.get("estrategia", "")])
n_dummy_nmar = len([d for d in decisoes_ausentes if "DUMMY" in d.get("estrategia", "")])
n_wins       = len([d for d in decisoes_outliers if d.get("estrategia") == "WINSORIZAR"])
n_flags_out  = len([d for d in decisoes_outliers if d.get("estrategia") == "FLAG"])

print(f"""
┌─────────────────────────────────────────────────────┐
│  RESUMO EXECUTIVO DO PIPELINE                       │
├─────────────────────────────────────────────────────┤
│  Variáveis removidas (alta ausência/constante): {n_removidas:4d} │
│  Variáveis imputadas:                           {n_imputadas:4d} │
│  Variáveis com dummy NMAR:                      {n_dummy_nmar:4d} │
│  Outliers: winsorizados:                        {n_wins:4d} │
│  Outliers: flagados (eventos reais):            {n_flags_out:4d} │
├─────────────────────────────────────────────────────┤
│  Base final: {df_fin.shape[0]:,} linhas × {df_fin.shape[1]} colunas{' ' * max(0, 17-len(str(df_fin.shape[1])))}│
│  Ausentes na base final: {df_fin.isna().sum().sum():,}{' ' * max(0, 27-len(str(df_fin.isna().sum().sum())))}│
└─────────────────────────────────────────────────────┘
""")

# Salva relatório em texto
relatorio_path = os.path.join(DIR_RELATORIOS, "relatorio_final_comparativo.txt")
with open(relatorio_path, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO COMPARATIVO — PIPELINE PNS 2019\n")
    f.write("=" * 60 + "\n\n")
    f.write(comparativo.to_string(index=False))
    f.write("\n\nDECISÕES DE IMPUTAÇÃO:\n")
    f.write(df_decisoes_ausentes.to_string(index=False))
    f.write("\n\nDECISÕES DE OUTLIERS:\n")
    f.write(df_decisoes_outliers.to_string(index=False))

print(f"[INFO] Relatório salvo: {relatorio_path}")
print(f"[INFO] Plots salvos em: {DIR_PLOTS}/")
print("\n[INFO] Pipeline concluído com sucesso.")