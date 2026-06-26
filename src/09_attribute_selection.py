"""
Script: 09_attribute_selection.py
Objetivo: Realizar a seleção de atributos (Etapa 7 do KDD - Critério 19) por meio de
          Entropia de Shannon e Ganho de Informação (Information Gain).
          Calcula a relevância de cada atributo em relação à classe alvo Q03001
          (Diabetes) e gera a base filtrada com as melhores características.

Input:
    data/processed/08_pns2019_outliers_treated.csv
Output:
    data/processed/09_pns2019_selected_features.csv
    docs/relatorios/relatorio_selecao_atributos.txt
    docs/relatorios/decisoes_selecao.csv
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_INPUT = os.path.join(BASE_DIR, "data", "processed", "08_pns2019_outliers_treated.csv")
PATH_OUTPUT_DATA = os.path.join(BASE_DIR, "data", "processed", "09_pns2019_selected_features.csv")
PATH_OUTPUT_DATA_NAMED = os.path.join(BASE_DIR, "data", "processed", "09_pns2019_selected_features_nomeados.csv")
PATH_REPORT = os.path.join(BASE_DIR, "docs", "relatorios", "relatorio_selecao_atributos.txt")
PATH_CSV_DECISÕES = os.path.join(BASE_DIR, "docs", "relatorios", "decisoes_selecao.csv")
PATH_TOP25_CSV = os.path.join(BASE_DIR, "docs", "relatorios", "tabela_top25_atributos_selecionados.csv")
PATH_DICT_JSON = os.path.join(BASE_DIR, "data", "first_atributes_dictionary_filter.json")
PATH_CLEAN_DICT_JSON = os.path.join(BASE_DIR, "data", "processed", "05_pns2019_clean_dictionary.json")

TARGET_COL = "Q03001"

# Vazamentos de dados (Data Leakage) e consequências pós-diagnóstico que devem ser excluídos
LEAKS = ["Q031", "Q02901", "J012", "Q00101", "J012_outlier_flag", "Q02901_outlier_flag"]  # Vazamentos/consequências

# Identificadores geográficos, redundantes ou ruídos demográficos/laborais sem nexo causal nutricional pro diabetes
EXCLUIR_MODELAGEM = [
    "V0001", "Calculo_IMC", "P027", "P04501", "P023", "P02401", "P02002", "P02102", "P02001", "P02101", "P02801", "P029", "P03201",
    "C011", "VDM001", "VDE014", "Nivel_Atividade_Fisica", "P039", "P038", "P040", "C009", "E017", "M005010", "VDF004",
    "P019",  # 47,2% NaN estrutural (pergunta condicional do IBGE) — inviável para modelagem
    "P036",  # ρ=0.80 com Minutos_Semanais_Exercicio (colinearidade severa)
]

def carregar_dicionarios():
    """Carrega descrições dos atributos a partir dos arquivos JSON e mapeamentos de derivadas."""
    descricoes = {}
    
    derivadas = {
        "Q03001": "Diagnóstico médico de diabetes (Variável Alvo / Target)",
        "Calculo_IMC": "Cálculo contínuo do Índice de Massa Corporal (IMC em kg/m²)",
        "IMC_Categoria": "Categorização do estado nutricional segundo o IMC (Baixo peso, Eutrofia, Sobrepeso, Obesidade)",
        "C008_Categoria": "Faixa etária do morador (Categorização da idade)",
        "Score_Ultraprocessados_Ontem": "Score de consumo de alimentos ultraprocessados no dia anterior",
        "Minutos_Semanais_Exercicio": "Tempo total semanal dedicado a exercícios físicos ou esportes (em minutos)",
        "Score_Saude_Mental": "Score indicativo de problemas de sono e saúde mental (derivado do Módulo N)",
        "Nivel_Atividade_Fisica": "Classificação do nível de atividade física (Ativo, Insuficientemente ativo, Inativo)",
        "Exposicao_Metabolica_Refrigerante": "Escore contínuo de exposição metabólica e química a refrigerantes (Dias × Peso Glicêmico/Tóxico)",
        "Classificacao_Consumo_Alcool": "Estratificação clínica de risco alcoólico segundo limites OMS e compulsão (Abstêmio, Baixo Risco, Abuso, Dependência)",
        "VDF003_outlier_flag": "[FLAG OUTLIER] Rendimento domiciliar per capita com valor extremo",
        "J012_outlier_flag": "[FLAG OUTLIER] Consultas médicas nos últimos 12 meses com valor extremo",
        "E017_outlier_flag": "[FLAG OUTLIER] Horas trabalhadas por semana com valor extremo",
        "P00103_outlier_flag": "[FLAG OUTLIER] Peso informado com valor extremo",
        "P00104_outlier_flag": "[FLAG OUTLIER] Peso final com valor extremo",
        "P00403_outlier_flag": "[FLAG OUTLIER] Altura informada com valor extremo",
        "P00404_outlier_flag": "[FLAG OUTLIER] Altura final com valor extremo"
    }
    descricoes.update(derivadas)
    
    if os.path.exists(PATH_CLEAN_DICT_JSON):
        try:
            with open(PATH_CLEAN_DICT_JSON, "r", encoding="utf-8") as f:
                descricoes.update(json.load(f))
        except Exception:
            pass
            
    if os.path.exists(PATH_DICT_JSON):
        try:
            with open(PATH_DICT_JSON, "r", encoding="utf-8") as f:
                descricoes.update(json.load(f))
        except Exception:
            pass
            
    return descricoes

def obter_descricao(col, descricoes_dict):
    if col in descricoes_dict:
        return descricoes_dict[col]
    elif col.endswith("_outlier_flag"):
        base_col = col.replace("_outlier_flag", "")
        if base_col in descricoes_dict:
            return f"[FLAG OUTLIER] {descricoes_dict[base_col]}"
        else:
            return "[FLAG OUTLIER] Variável com valor extremo"
    return "Descrição não encontrada no dicionário"

def shannon_entropy(series):
    """Calcula a Entropia de Shannon H(Y) de uma variável discreta."""
    total = len(series)
    if total == 0:
        return 0.0
    counts = series.value_counts()
    probabilities = counts / total
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def information_gain(df, feature, target):
    """Calcula o Ganho de Informação IG(Y, X) = H(Y) - H(Y|X)."""
    # 1. Entropia total da classe alvo H(Y)
    h_y = shannon_entropy(df[target])
    
    # 2. Entropia condicional H(Y|X)
    h_y_x = 0.0
    total_instancias = len(df)
    
    # Para atributos contínuos com muitos valores únicos, fazemos uma discretização temporária
    # em 5 bins usando quantis (qcut) para evitar o sobreajuste clássico do Ganho de Informação.
    if df[feature].dtype in [np.float64, np.int64] and df[feature].nunique() > 10:
        try:
            # pd.qcut divide os dados em intervalos com frequências iguais
            valores_calculo = pd.qcut(df[feature], q=5, labels=False, duplicates='drop')
        except Exception:
            # Fallback caso dê erro por excesso de valores idênticos
            valores_calculo = pd.cut(df[feature], bins=5, labels=False)
    else:
        valores_calculo = df[feature]
    
    # Agrupa a classe alvo pelas categorias do atributo avaliado (incluindo NaNs como categoria estrutural)
    grupos = df[target].groupby(valores_calculo, dropna=False)
    for val, grupo in grupos:
        p_x = len(grupo) / total_instancias
        h_grupo = shannon_entropy(grupo)
        h_y_x += p_x * h_grupo
        
    # Ganho de Informação
    ig = h_y - h_y_x
    return ig

def main():
    print("=" * 70)
    print("  ETAPA 7 — SELEÇÃO DE ATRIBUTOS POR ENTROPIA | PNS 2019")
    print("=" * 70)
    
    if not os.path.exists(PATH_INPUT):
        print(f"[ERRO] Arquivo de entrada não encontrado: {PATH_INPUT}")
        return
        
    df = pd.read_csv(PATH_INPUT)
    print(f"[INFO] Dataset carregado: {df.shape[0]:,} registros e {df.shape[1]} colunas.")
    
    # Calcular Entropia do Target
    h_target = shannon_entropy(df[TARGET_COL])
    print(f"[INFO] Entropia de Shannon do Target H({TARGET_COL}) = {h_target:.5f}")
    
    # Identificar colunas elegíveis
    colunas_analise = [
        col for col in df.columns 
        if col != TARGET_COL and col not in LEAKS and col not in EXCLUIR_MODELAGEM
    ]
    
    print(f"[INFO] Analisando {len(colunas_analise)} atributos...")
    descricoes_dict = carregar_dicionarios()
    
    # Calcular Ganho de Informação para cada atributo
    resultados = []
    for col in colunas_analise:
        ig = information_gain(df, col, TARGET_COL)
        resultados.append({
            "atributo": col,
            "descricao": obter_descricao(col, descricoes_dict),
            "ganho_informacao": ig,
            "tipo": str(df[col].dtype),
            "valores_unicos": df[col].nunique()
        })
        
    # Ordenar por Ganho de Informação decrescente
    df_resultados = pd.DataFrame(resultados).sort_values(by="ganho_informacao", ascending=False)
    
    # Salvar tabela de decisões em CSV
    df_resultados.to_csv(PATH_CSV_DECISÕES, index=False, encoding="utf-8")
    print(f"[SALVO] Ranking completo de entropia salvo em: {PATH_CSV_DECISÕES}")
    
    # Definir critério de corte:
    # 1. Podemos filtrar por um threshold de relevância (ex: IG > 0.0005)
    # 2. Ou selecionar os TOP N atributos mais informativos (ex: Top 19)
    # Para o estudo de diabetes e hábitos nutricionais, selecionaremos no máximo 19 atributos (restrição de desbalanceamento + remoção de colinearidade)
    TOP_N = 19
    df_top_n = df_resultados.head(TOP_N).copy()
    df_top_n.insert(0, "ranking", range(1, TOP_N + 1))
    atributos_selecionados = df_top_n["atributo"].tolist()
    
    # Salvar tabela específica dos Top 25 selecionados com descrição
    df_top_n.to_csv(PATH_TOP25_CSV, index=False, encoding="utf-8")
    print(f"[SALVO] Tabela dos Top {TOP_N} atributos selecionados salva em: {PATH_TOP25_CSV}")
    
    print(f"\n[SELEÇÃO] Top {TOP_N} atributos selecionados por Entropia:")
    for idx, row in df_top_n.iterrows():
        print(f"  {row['ranking']:02d}. {row['atributo']:<25} | IG: {row['ganho_informacao']:.6f} | {row['descricao']}")
        
    # Criar a nova base de dados apenas com os atributos selecionados + o Target
    colunas_finais = atributos_selecionados + [TARGET_COL]
    df_selecionado = df[colunas_finais]
    df_selecionado.to_csv(PATH_OUTPUT_DATA, index=False, encoding="utf-8")
    print(f"\n[SALVO] Base filtrada salva em: {PATH_OUTPUT_DATA}")
    
    # Exportar também a base filtrada com os nomes descritivos nas colunas
    mapeamento_nomes = {col: f"{col} - {obter_descricao(col, descricoes_dict)}" for col in colunas_finais}
    df_selecionado_named = df_selecionado.rename(columns=mapeamento_nomes)
    df_selecionado_named.to_csv(PATH_OUTPUT_DATA_NAMED, index=False, encoding="utf-8")
    print(f"[SALVO] Base filtrada com nomes descritivos salva em: {PATH_OUTPUT_DATA_NAMED}")
    print(f"[INFO] Dimensões da base final: {df_selecionado.shape[0]:,} linhas × {df_selecionado.shape[1]} colunas.")
    
    # Gerar Relatório Textual para o Artigo
    with open(PATH_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  RELATÓRIO DE SELEÇÃO DE ATRIBUTOS POR ENTROPIA (IG) | PNS 2019\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"1. INFORMAÇÕES DA BASE\n")
        f.write(f"   - Base de entrada: {df.shape[0]:,} instâncias, {df.shape[1]} colunas\n")
        f.write(f"   - Variável Alvo (Target): {TARGET_COL}\n")
        f.write(f"   - Entropia do Target H({TARGET_COL}): {h_target:.6f}\n")
        f.write(f"   - Exclusões por Vazamento (Leakage): {LEAKS}\n")
        f.write(f"   - Exclusões Identificadoras: {EXCLUIR_MODELAGEM}\n\n")
        
        f.write(f"2. CRITÉRIO DE FILTRAGEM\n")
        f.write(f"   - Método: Seleção por Limiar de Ganho de Informação (Entropia de Shannon)\n")
        f.write(f"   - Decisão: Retenção dos Top {TOP_N} atributos mais informativos para a modelagem\n")
        f.write(f"   - Base resultante: {df_selecionado.shape[0]:,} instâncias × {df_selecionado.shape[1]} colunas\n\n")
        
        f.write(f"3. RANKING COMPLETO DE ATRIBUTOS (GANHO DE INFORMAÇÃO)\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Ranking':<8} {'Atributo':<22} {'Ganho de Informação':<22} {'Uniques':<10} {'Status':<14} {'Descrição'}\n")
        f.write("-" * 110 + "\n")
        
        for idx, (_, row) in enumerate(df_resultados.iterrows(), 1):
            status = "[Selecionado]" if row['atributo'] in atributos_selecionados else "[Descartado]"
            desc_completa = str(row['descricao'])
            f.write(f"{idx:<8} {row['atributo']:<22} {row['ganho_informacao']:<22.6f} {row['valores_unicos']:<10} {status:<14} {desc_completa}\n")
            
        f.write("\n\n4. JUSTIFICATIVA CLÍNICA E COMPORTAMENTAL DOS TOP ATRIBUTOS\n")
        f.write("-" * 80 + "\n")
        f.write("Os atributos no topo do ranking revelam correlações importantes com a ocorrência de diabetes:\n")
        f.write("- Atributos clínicos e de monitoramento (ex: aferição de pressão arterial Q00101) apresentam forte correlação com diabéticos diagnosticados.\n")
        f.write("- Indicadores antropométricos (IMC e Peso) e faixa etária (C008_Categoria) apresentam forte ganho de informação, condizente com a fisiopatologia do Diabetes Tipo 2.\n")
        f.write("- Variáveis de consumo alimentar e estilo de vida (como o Score_Ultraprocessados_Ontem e Nivel_Atividade_Fisica) representam os fatores de risco comportamentais de interesse do estudo.\n")

    print(f"[SALVO] Relatório textual salvo em: {PATH_REPORT}")

if __name__ == "__main__":
    main()
